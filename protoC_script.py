import os
from shutil import move
import argparse
import sys


def main(arguments):

    # update pip
    os.system("python -m pip install --upgrade pip")

    # Compile .proto to python files

    current_dir = os.getcwd()

    os.chdir('../')
    src_dir = os.getcwd()
    os.chdir(current_dir)
    
    #folder to insert files
    config_folder = os.sep + "lib" + os.sep + "src" + os.sep + "owlracer"
    

    # Because of cmd input the strings are in need
    var = "python -m grpc_tools.protoc " + "-I " + src_dir + os.sep + "Protobuf " + src_dir + os.sep + "Protobuf" + os.sep + "matlabs.owlracer.core.proto " + "--python_out=" + current_dir + " --grpc_python_out=" + current_dir

    ### test ###
    testVar1 = "python -m grpc_tools.protoc " + "-I " + src_dir + os.sep + "Protobuf " + src_dir + os.sep + "Protobuf" + os.sep + "matlabs.owlracer.core.proto " + "--python_out=" + current_dir + config_folder + " --grpc_python_out=" + current_dir + config_folder
    ############

    os.system(
        "python -m grpc_tools.protoc "
        "-I " + src_dir + os.sep + "Protobuf " +
        src_dir + os.sep + "Protobuf" + os.sep + "matlabs.owlracer.core.proto "
        "--python_out=" + current_dir + config_folder +
        " --grpc_python_out=" + current_dir + config_folder)

    #Rename the exported file to the right format

    renamed_grpc_file = current_dir + config_folder + os.sep + "core_pb2_grpc.py"
    grpc_file = current_dir + config_folder + os.sep + "matlabs.owlracer.core_pb2_grpc.py"

    if os.path.exists(renamed_grpc_file):
        os.remove(renamed_grpc_file)

    if os.path.exists(grpc_file):
        os.rename(grpc_file, renamed_grpc_file)

    try:
        #moves the file in the right directory for the lib
        move(renamed_grpc_file, current_dir + config_folder + os.sep + "grpcClient" + os.sep + "core_pb2_grpc.py")
    except Exception as ex:
        print(ex)

    #change dir and build pack

    os.chdir('./lib')
    os.system("python -m build")
    os.chdir('../')

    #install requirements depending on arguments
    if arguments.dev:
        os.system("python -m pip install -r ./requirements_dev.txt")
    else:
        os.system("python -m pip install -r ./requirements.txt")


def parse_arguments(args: list[str]):
    parser = argparse.ArgumentParser(
            description="Decide which requirements are to be installed",
        )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Install development requirements for training models",
        default=False
    )
    return parser.parse_args(args)


if __name__ == '__main__':
    arguments = parse_arguments(sys.argv[1:])
    main(arguments)
