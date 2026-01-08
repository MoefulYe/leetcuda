{
  description = "LeetCUDA CUDA dev environment (NixOS + uv venv)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      pythonPackages = pkgs.python312Packages;
      python = pythonPackages.python;
      pybind11 = pythonPackages.pybind11;
      cudaPackages = pkgs.cudaPackages_12_8;
      llvm = pkgs.llvmPackages_21;
      clang-tools = llvm.clang-tools;
      stdenv = llvm.stdenv;
      fhs = pkgs.buildFHSEnv {
        name = "leetcuda-fhs";
        targetPkgs = pkgs': [ pkgs'.glibc ];
        runScript = "bash";
      };
      nsys = cudaPackages.nsight_systems;
      ncu = cudaPackages.nsight_compute.overrideAttrs (old: {
        postInstall = old.postInstall + ''
          ln -s $out/bin/target/linux-desktop-glibc_2_11_3-x64 \
            $out/bin/target/linux-desktop-glibc_2_11_3-x86
          ln -s $out/sections $out/bin/sections
        '';

        meta.description = "";
      });
    in
    {
      formatter.${system} = pkgs.alejandra;
      devShells.${system}.default =
        pkgs.mkShell.override
          {
            inherit stdenv;
          }
          {
            name = "leetcuda";
            packages = with pkgs; [
              fhs
              cudaPackages.cudatoolkit
              ncu
              nsys
              python
              pybind11
              uv
              cmake
              ninja
              pkg-config
              clang-tools
              git
            ];

            shellHook =
              let
                ld_libpath = pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.glibc
                ];
              in
              ''
                export DISABLE_DIRENV=1

                # Speed up PyTorch extension compilation on RTX 3080 (Ampere, sm_86).
                export TORCH_CUDA_ARCH_LIST="8.6"

                # CUDA toolkit path for build tooling (nvcc, headers, etc.)
                export CUDA_HOME=${cudaPackages.cudatoolkit}
                export CUDA_PATH=${cudaPackages.cudatoolkit}
                export CUDACXX=${cudaPackages.cudatoolkit}/bin/nvcc

                # Keep LD_LIBRARY_PATH to avoid runtime "cannot open shared object file".
                # On NixOS, NVIDIA driver libs are typically exposed via /run/opengl-driver/lib.
                driver_lib="/run/opengl-driver/lib"
                driver_lib32="/run/opengl-driver-32/lib"


                export LD_LIBRARY_PATH="$driver_lib:$driver_lib32:${ld_libpath}:/nix/store/xx7cm72qy2c0643cm1ipngd87aqwkcdp-glibc-2.40-66/lib/"

                # Create & activate a local venv via uv (install torch wheels inside it).
                if [ ! -d ".venv" ]; then
                  uv venv
                fi
                source .venv/bin/activate
                export PATH="''${PWD}/.venv/bin:''${PATH}"
                export TORCH_CUDA_ARCH_LIST="8.6"

                # Build a real ':'-separated PYTHONPATH (expand build/kernels/*/).
                PYTHONPATH="''${PWD}/common:''${PYTHONPATH:-}"
                for d in "''${PWD}"/build/kernels/*/; do
                  [ -d "$d" ] || continue
                  PYTHONPATH="$PYTHONPATH:$d"
                done
                export PYTHONPATH
              '';
          };
    };
}
