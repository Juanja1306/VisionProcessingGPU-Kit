import os
import platform

def init_cuda():
    """
    Configures CUDA environment variables based on the operating system.
    """
    system = platform.system()
    
    if system == "Windows":
        # # Visual Studio environment variables for nvcc on Windows
        # vs_compiler_path = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\bin\Hostx64\x64"

        # if vs_compiler_path not in os.environ.get("PATH", ""):
        #     os.environ["PATH"] = os.environ.get("PATH", "") + ";" + vs_compiler_path

        # vs_include = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\include;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\ATLMFC\include;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\VS\include;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\ucrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\um;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\shared;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\winrt;C:\Program Files (x86)\Windows Kits\10\include\10.0.26100.0\cppwinrt"

        # os.environ["INCLUDE"] = vs_include

        # vs_lib = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\ATLMFC\lib\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.42.34433\lib\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\ucrt\x64;C:\Program Files (x86)\Windows Kits\10\lib\10.0.26100.0\um\x64"

        # os.environ["LIB"] = vs_lib
        
        print("CUDA environment configured for Windows (Visual Studio paths set).")

    elif system == "Linux":
        # On Linux (Docker), we assume the base image (nvidia/cuda) provides the necessary environment.
        # We might need to add /usr/local/cuda/bin to PATH if it's not there.
        cuda_bin = "/usr/local/cuda/bin"
        if cuda_bin not in os.environ.get("PATH", ""):
             os.environ["PATH"] = cuda_bin + ":" + os.environ.get("PATH", "")
        
        print("CUDA environment configured for Linux.")
    
    else:
        print(f"Warning: Unknown operating system '{system}'. CUDA might not be configured correctly.")
