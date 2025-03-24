# jobshop-gpu

This is a repository for computing the Flexible Job Shop Scheduling Problem on a GPU

## Compilation and running

CMake is used for project compilation.
To compile the project:

```bash
mkdir build
cd build
cmake ..
make
```

To run the project:

```bash
cd build
./JobShopScheduler
```

CMake and CMake Tools extensions for VSCode are suggested.

## CUDA Setup

This may make you lose your sanity, so brace yourself. You should delete all Nvidia packages before proceeding with the CUDA Toolkit installation. Keep deleting them until the prompt for existing Nvidia packages stops popping up.

1. Install required packages

    ```bash
    sudo apt update
    sudo apt install -y build-essential libglvnd-dev pkg-config
    ```

2. Delete all nvidia packages and drivers

    ```bash
    sudo apt remove --purge "nvidia*"
    dpkg -l | grep nvidia
    sudo apt purge ...
    ```

    Delete all packages found with this command and additionally, purge libnvidia packages found on your PC. Reboot after doing the above.

3. Disable the `nouveau` driver. This will probably mess up your screen resolution, but it will be fine upon CUDA Toolkit installation.

    ```bash
    echo "blacklist nouveau" | sudo tee /etc/modprobe.d/blacklist-nouveau.conf
    echo "options nouveau modeset=0" | sudo tee -a /etc/modprobe.d/blacklist-nouveau.conf
    sudo update-initramfs -u
    sudo reboot
    ```

    After that, check if NVIDIA drivers are successfully uninstalled: `nvidia-smi`.  This should say `command not found`.

4. Download the CUDA Toolkit from here: <https://developer.nvidia.com/cuda-toolkit>

    Then, `chmod +x cuda` + [TAB]. Next: `sudo ./cuda` + [TAB]

    Choose the `fs` package if you want to use clusters for computations.

5. Check if CUDA works `nvcc --version`

6. Add these lines to `bashrc` or `zshrc`

    ```bash
    export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    source ~/.bashrc
    ```

7. You should be all set after this step, congratulations.
