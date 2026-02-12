{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  buildInputs = with pkgs; [
    nodejs_22
    pkg-config
    libglvnd
    xorg.libX11
    xorg.libXi
    xorg.libXext
  ];

  shellHook = ''
    export LD_LIBRARY_PATH="${
      pkgs.lib.makeLibraryPath [
        pkgs.libglvnd
        pkgs.xorg.libX11
        pkgs.xorg.libXi
        pkgs.xorg.libXext
      ]
    }:$LD_LIBRARY_PATH"
    # Create venv if it doesn't exist
    if [ ! -d "venv" ]; then
      python -m venv venv
    fi

    # Source the venv
    source venv/bin/activate
  '';
}
