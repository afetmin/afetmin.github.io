#!/usr/bin/env bash
#
# Run jekyll serve and then launch the site

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUBY31_BIN="/opt/homebrew/opt/ruby@3.1/bin"

export PATH="$RUBY31_BIN:$PATH"
export BUNDLE_PATH="$PROJECT_DIR/vendor/bundle"
export BUNDLE_DISABLE_SHARED_GEMS=true
export GEM_HOME="$PROJECT_DIR/vendor/bundle"
export GEM_PATH="$PROJECT_DIR/vendor/bundle"
export BUNDLE_USER_HOME="$PROJECT_DIR/.bundle-home"
export BUNDLE_APP_CONFIG="$PROJECT_DIR/.bundle"
export BUNDLE_CACHE_PATH="$PROJECT_DIR/.bundle-cache"

prod=false
command="bundle exec jekyll s -l"
host="127.0.0.1"

help() {
  echo "Usage:"
  echo
  echo "   bash /path/to/run [options]"
  echo
  echo "Options:"
  echo "     -H, --host [HOST]    Host to bind to."
  echo "     -p, --production     Run Jekyll in 'production' mode."
  echo "     -h, --help           Print this help information."
}

while (($#)); do
  opt="$1"
  case $opt in
  -H | --host)
    host="$2"
    shift 2
    ;;
  -p | --production)
    prod=true
    shift
    ;;
  -h | --help)
    help
    exit 0
    ;;
  *)
    echo -e "> Unknown option: '$opt'\n"
    help
    exit 1
    ;;
  esac
done

command="$command -H $host"

if $prod; then
  command="JEKYLL_ENV=production $command"
fi

if [ -e /proc/1/cgroup ] && grep -q docker /proc/1/cgroup; then
  command="$command --force_polling"
fi

echo -e "\n> $command\n"
eval "$command"
