import os
from shutil import move

def main():

    # Compile .proto to python files

    # install requirements
    os.system("python -m pip install -r ./requirements.txt")

    current_dir = os.getcwd()

    os.chdir('../')
    src_dir = os.getcwd()
    os.chdir(current_dir)
    
    #folder to insert files
    config_folder = os.sep + "lib" + os.sep + "src" + os.sep + "owlracer"
    

    # Because of cmd input the strings are in neet 
    #var = "python3 -m grpc_tools.protoc " + "-I " + src_dir + os.sep + "Protobuf " + src_dir + os.sep + "Protobuf" + os.sep + "matlabs.owlracer.core.proto " + "--python_out=" + current_dir + " --grpc_python_out=" + current_dir
    
    os.system(
        "python -m grpc_tools.protoc "
        "-I " + src_dir + os.sep + "Protobuf " +
        src_dir + os.sep + "Protobuf" + os.sep + "matlabs.owlracer.core.proto "
        "--python_out=" + current_dir + config_folder +
        " --grpc_python_out=" + current_dir + config_folder )

    
    #Rename the exported file to the right format

    renamed_grpc_file = current_dir + config_folder + os.sep + "core_pb2_grpc.py"
    grpc_file = current_dir + config_folder + os.sep + "matlabs.owlracer.core_pb2_grpc.py"

    if os.path.exists(renamed_grpc_file):
        os.remove(renamed_grpc_file)

    if os.path.exists(grpc_file):
        os.rename(grpc_file, renamed_grpc_file)


    #moves the file in the right directory for the lib
    #move(renamed_grpc_file, current_dir + config_folder + os.sep + "grpcClient" + os.sep + "core_pb2_grpc.py")

    #change dir and build pack
    os.chdir('./lib')
    os.system("python -m build")
    os.chdir('../')

    #install requirements
    os.system("python -m pip install -r ./requirements.txt")


if __name__ == '__main__':
    main()
