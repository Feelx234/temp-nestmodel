
import subprocess
from pathlib import Path
import os
from tnestmodel.datasets import datasets


base = r"https://raw.githubusercontent.com/busyweaver/temporal_nest/main/datasets/networks/"


suffixes = [
    dataset.name+".csv" for dataset in datasets
]

all_links = [base+suff for suff in suffixes]

def download_datasets():

    print("filepath\t", Path(__file__).resolve())
    print("file_folder\t", Path(__file__).parent.absolute())


    parent = Path(__file__).parent.absolute()

    if str(parent.name) == "scripts":
        parent = parent.parent
    if str(parent.name) == "tnestmodel":
        parent = parent.parent
    if str(parent.name) == "src":
        parent = parent.parent
    if str(parent.name)!="datasets":
        parent = parent/"datasets"

    if not parent.is_dir():
        print("creating ", parent)
        parent.mkdir(exist_ok=True)
    print("dataset_folder\t", parent)

    #assert parent.exists(), str(parent)
    links = all_links
    download_names = suffixes
    final_files = download_names
    #final_files = [("cit-HepPh.txt",), ("ca-AstroPh.txt",), networkscience_files, ("web-Google.txt",), ("soc-pokec-relationships.txt",)]


    if os.name == 'nt':
        # if you want to know more about the download command for windows use
        # https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/invoke-webrequest?view=powershell-7.3
        def download_command_windows(link, parent, file):
            return "powershell wget " +"-Uri " + link +" -OutFile " + str(parent/file)
        dowload_command = download_command_windows

        def unzip_command_windows(parent, file):
            return "unzip "  + str(parent/file) + " -d " + str((parent/file).stem)
        unzip_command = unzip_command_windows
    else:
        def download_command_linux(link, parent, file):
            return "wget " + link +" -P " + str(parent)
        dowload_command = download_command_linux

        def unzip_command_linux(link, parent, file):
            return "unzip " + str(file)
        unzip_command = unzip_command_linux


    files = list(file.name for file in parent.iterdir())
    for link, download_name, finals in zip(links, download_names, final_files):
        if all(final in files for final in finals):
            continue

        if download_name in files:
            print(f"{download_name} is present")
        else:
            # download file
            command = dowload_command(link, parent, download_name)
            print()
            print("<<< downloading " + download_name)
            subprocess.call(command, shell=True, cwd=str(parent))

        if download_name.endswith(".gz"):
            command = "gzip -d " + str(download_name)
            print("<<< extracting " + download_name)
            subprocess.call(command, shell=True , cwd=str(parent))

        if download_name.endswith(".zip"):
            command = unzip_command(parent, download_name)
            print("<<< extracting " + download_name)
            subprocess.call(command, shell=True , cwd=str(parent))

    print()
    print("done")
    print()
#dataset_path_file = Path(__file__).parent.absolute()/"datasets_path.txt"
#if not dataset_path_file.is_file():
#    with open(dataset_path_file, "w") as f:
#        f.write(str(parent))

if __name__ == "__main__":
    download_datasets()