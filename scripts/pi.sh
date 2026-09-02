#!/bin/sh
set -eu

project=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
custom_config=${PI_CODING_AGENT_DIR:-}
config=${custom_config:-"$project/build/pi-qwen3x"}

command -v node >/dev/null 2>&1 || {
    echo "pi requires Node >= 22.19" >&2
    exit 1
}
node -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22 || a===22&&b>=19 ? 0 : 1)' || {
    echo "pi requires Node >= 22.19; current: $(node --version)" >&2
    exit 1
}
command -v pi >/dev/null 2>&1 || {
    echo "pi is not installed" >&2
    exit 1
}
curl -fsS http://127.0.0.1:8000/readyz >/dev/null || {
    echo "qwen3x is not ready; run: make -C $project serve-4b" >&2
    exit 1
}

mkdir -p "$config"
if [ -z "$custom_config" ] || [ ! -f "$config/models.json" ]; then
    cp "$project/scripts/pi-models.json" "$config/models.json"
fi
if [ -z "$custom_config" ] || [ ! -f "$config/settings.json" ]; then
    node - "$config/settings.json" "$project/scripts/pi-settings.json" <<'NODE'
const fs = require("fs");
const [target, defaults] = process.argv.slice(2);
const settings = fs.existsSync(target)
    ? JSON.parse(fs.readFileSync(target, "utf8")) : {};
settings.compaction = JSON.parse(
    fs.readFileSync(defaults, "utf8")).compaction;
fs.writeFileSync(target, JSON.stringify(settings, null, 2) + "\n", {
    mode: 0o600,
});
NODE
fi

export PI_CODING_AGENT_DIR="$config"
exec pi --provider qwen3x --model qwen3.5-4b "$@"
