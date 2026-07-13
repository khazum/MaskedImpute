#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
manifest="$repo_root/environments/saver-r.lock.json"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 LIBRARY_DIR PINNED_SAVER_SOURCE [BUILD_RECEIPT]" >&2
  exit 64
fi

library_dir=$(realpath -m "$1")
saver_source=$(realpath "$2")
build_receipt=${3:-"${library_dir}.build-receipt.json"}
build_receipt=$(realpath -m "$build_receipt")

case "$library_dir" in
  /tmp/*|"$repo_root"/artifacts/*) ;;
  *)
    echo "SAVER library must be under /tmp or the ignored artifacts directory" >&2
    exit 64
    ;;
esac
if [[ -e "$library_dir" ]]; then
  echo "refusing to replace existing SAVER library: $library_dir" >&2
  exit 73
fi

for command in awk base64 curl git jq R realpath Rscript sha256sum; do
  command -v "$command" >/dev/null || {
    echo "required build command is unavailable: $command" >&2
    exit 69
  }
done

hash_saver_library() {
  local root=$1
  local unsupported
  unsupported=$(find "$root" -mindepth 1 \( -type l -o \! -type d -a \! -type f \) -print -quit)
  if [[ -n "$unsupported" ]]; then
    echo "unsupported installed-library entry: $unsupported" >&2
    return 65
  fi
  while IFS= read -r relative; do
    case "$relative" in
      *[!abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.+@/:=\-]*)
        echo "unsupported installed-library path: $relative" >&2
        return 65
        ;;
    esac
  done < <(find "$root" -type f -printf '%P\n' | LC_ALL=C sort)
  (
    cd "$root"
    while IFS= read -r relative; do
      sha256sum -- "$relative"
    done < <(find . -type f -printf '%P\n' | LC_ALL=C sort)
  ) | sha256sum | awk '{print $1}'
}

expected_r=$(jq -er '.r_version' "$manifest")
expected_library_sha=$(jq -er '.installed_library_sha256' "$manifest")
actual_r=$(Rscript --vanilla -e 'cat(paste(R.version$major, R.version$minor, sep="."))')
if [[ "$actual_r" != "$expected_r" ]]; then
  echo "R version mismatch: expected $expected_r, observed $actual_r" >&2
  exit 65
fi

expected_revision=$(jq -er '.upstream_saver.revision' "$manifest")
expected_tree=$(jq -er '.upstream_saver.tree' "$manifest")
expected_url=$(jq -er '.upstream_saver.url' "$manifest")
actual_revision=$(git -C "$saver_source" rev-parse HEAD)
actual_tree=$(git -C "$saver_source" rev-parse 'HEAD^{tree}')
actual_url=$(git -C "$saver_source" remote get-url origin)
actual_status=$(git -C "$saver_source" status --porcelain=v1 --untracked-files=all)
if [[ "$actual_revision" != "$expected_revision" ||
      "$actual_tree" != "$expected_tree" ||
      "$actual_url" != "$expected_url" ||
      -n "$actual_status" ]]; then
  echo "pinned SAVER source identity or pristine status differs from lock" >&2
  exit 65
fi

library_parent=$(dirname "$library_dir")
mkdir -p "$library_parent" "$(dirname "$build_receipt")"
stage=$(mktemp -d "$library_parent/.saver-r-build.XXXXXX")
cleanup() {
  if [[ ${build_succeeded:-false} == true ]]; then
    rm -rf -- "$stage"
  else
    echo "failed build stage preserved at $stage" >&2
  fi
}
trap cleanup EXIT
downloads="$stage/downloads"
stage_library="$stage/library"
mkdir -p "$downloads" "$stage_library"

while IFS= read -r encoded; do
  package_json=$(printf '%s' "$encoded" | base64 --decode)
  package=$(jq -er '.package' <<<"$package_json")
  version=$(jq -er '.version' <<<"$package_json")
  url=$(jq -er '.url' <<<"$package_json")
  expected_sha=$(jq -er '.sha256' <<<"$package_json")
  archive="$downloads/${package}_${version}.tar.gz"
  curl --fail --location --retry 3 --silent --show-error "$url" --output "$archive"
  printf '%s  %s\n' "$expected_sha" "$archive" | sha256sum --check --status
  MAKEFLAGS=-j1 \
    R_LIBS="$stage_library" \
    R_LIBS_SITE="$stage_library" \
    R_LIBS_USER="$stage_library" \
    R CMD INSTALL --no-multiarch --library="$stage_library" "$archive"
done < <(jq -r '.packages[] | @base64' "$manifest")

MAKEFLAGS=-j1 \
  R_LIBS="$stage_library" \
  R_LIBS_SITE="$stage_library" \
  R_LIBS_USER="$stage_library" \
  R CMD INSTALL --no-multiarch --library="$stage_library" "$saver_source"

expected_versions="$stage/expected-versions.tsv"
jq -r '.packages[] | [.package,.version] | @tsv' "$manifest" >"$expected_versions"
jq -r '[.upstream_saver.package,.upstream_saver.version] | @tsv' "$manifest" >>"$expected_versions"
Rscript --vanilla - "$stage_library" "$expected_versions" <<'RSCRIPT'
args <- commandArgs(trailingOnly=TRUE)
library_dir <- normalizePath(args[[1]], mustWork=TRUE)
expected <- read.delim(args[[2]], header=FALSE, col.names=c("package", "version"),
                       stringsAsFactors=FALSE)
.libPaths(c(library_dir, .Library))
for (index in seq_len(nrow(expected))) {
  package <- expected$package[[index]]
  version <- expected$version[[index]]
  path <- normalizePath(find.package(package, lib.loc=library_dir), mustWork=TRUE)
  if (!identical(dirname(path), library_dir)) stop("package escaped locked library")
  observed <- as.character(packageDescription(
    package, fields="Version", lib.loc=library_dir
  ))
  if (!identical(observed, version)) {
    stop(paste("package version mismatch", package))
  }
}
RSCRIPT

observed_library_sha=$(hash_saver_library "$stage_library")
if [[ "$observed_library_sha" != "$expected_library_sha" ]]; then
  echo "installed SAVER library digest mismatch: expected $expected_library_sha, observed $observed_library_sha" >&2
  exit 65
fi

post_revision=$(git -C "$saver_source" rev-parse HEAD)
post_tree=$(git -C "$saver_source" rev-parse 'HEAD^{tree}')
post_url=$(git -C "$saver_source" remote get-url origin)
post_status=$(git -C "$saver_source" status --porcelain=v1 --untracked-files=all)
if [[ "$post_revision" != "$expected_revision" ||
      "$post_tree" != "$expected_tree" ||
      "$post_url" != "$expected_url" ||
      -n "$post_status" ]]; then
  echo "pinned SAVER source changed during environment build" >&2
  exit 65
fi

mv "$stage_library" "$library_dir"
manifest_sha=$(sha256sum "$manifest" | awk '{print $1}')
package_versions=$(jq -c '
  reduce .packages[] as $item ({}; .[$item.package]=$item.version)
  + {SAVER:.upstream_saver.version}
' "$manifest")
jq -n \
  --arg status build_complete \
  --arg manifest_sha256 "$manifest_sha" \
  --arg r_version "$actual_r" \
  --arg library_dir "$library_dir" \
  --arg installed_library_sha256 "$observed_library_sha" \
  --arg source_revision "$actual_revision" \
  --argjson package_versions "$package_versions" \
  '{
    schema_version:1,
    status:$status,
    manifest_sha256:$manifest_sha256,
    r_version:$r_version,
    library_dir:$library_dir,
    installed_library_sha256:$installed_library_sha256,
    saver_source_revision:$source_revision,
    package_versions:$package_versions
  }' >"$build_receipt"

trap - EXIT
build_succeeded=true
rm -rf -- "$stage"
echo "SAVER R library built at $library_dir"
echo "Build receipt written to $build_receipt"
