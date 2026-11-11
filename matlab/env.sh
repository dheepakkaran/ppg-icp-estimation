# Optional: use a local tools folder for pixi if it exists (no sudo needed)
if [ -d "$HOME/workspace/.tools/bin" ]; then
  export PIXI_HOME=$HOME/workspace/.tools/pixi
  export PIXI_CACHE_DIR=$HOME/workspace/.tools/pixi-cache
  export RATTLER_CACHE_DIR=$HOME/workspace/.tools/pixi-cache
  export PATH=$HOME/workspace/.tools/bin:$PATH
fi
