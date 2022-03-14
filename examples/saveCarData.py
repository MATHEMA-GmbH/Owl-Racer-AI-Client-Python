import pandas as pd

#API import
from matlabs.owlracer import core_pb2
from grpcClient import core_pb2_grpc
from serviceClient import service, commands


def mainLoop():
  
    env = service.Env(spectator=True)
    
    carID = env.getCarIDs()
    carID = core_pb2.GuidData(guidString = str(carID.guids[0]).split("\"")[1])
    #print("On the server is {} car with IDs".format())
    env.setCarID(carID=carID)
    columList = ["MapNr","pX","pY","rotation","maxVelocity","acceleration","velocity","isCrashed","isDone","scoreStep","scoreOverall","ticks","dFront","dFrontL","dFrontR","dLeft","dRight","checkPoint"]

    #loop variables
    sampleNr=0
    mapNr = 0
    dataList = []
    lastTick = -1

    while(sampleNr<10):

        step_result: core_pb2_grpc.RaceCarData = env.getCarData()
        if step_result.ticks > lastTick:
            dataList.append([mapNr,step_result.position.x, step_result.position.y, step_result.rotation, step_result.maxVelocity, step_result.acceleration, step_result.velocity, step_result.isCrashed, step_result.isDone, step_result.scoreStep, step_result.scoreOverall, step_result.ticks,
                        step_result.distance.front, step_result.distance.frontLeft, step_result.distance.frontRight, step_result.distance.left, step_result.distance.right, step_result.checkPoint])

            #print("Car Pos: {} {}, Vel: {} forward distance {}".format(step_result.position.x, step_result.position.y, step_result.velocity, step_result.distance.front))
            sampleNr += 1
        lastTick = step_result.ticks


    dataFrame = pd.DataFrame(dataList, columns=columList)
    print("Done sampling")
    filename = "../sampling0.csv"
    dataFrame.to_csv(filename)
    print("Saved data tp csv with file name {}".format(filename))

if __name__ == '__main__':
    mainLoop()
