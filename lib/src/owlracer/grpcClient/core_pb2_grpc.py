# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from google.protobuf import empty_pb2 as google_dot_protobuf_dot_empty__pb2
from matlabs.owlracer import core_pb2 as matlabs_dot_owlracer_dot_core__pb2


class GrpcCoreServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetCarIds = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcCoreService/GetCarIds',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
                response_deserializer=matlabs_dot_owlracer_dot_core__pb2.GuidListData.FromString,
                )
        self.CreateSession = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcCoreService/CreateSession',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.CreateSessionData.SerializeToString,
                response_deserializer=matlabs_dot_owlracer_dot_core__pb2.SessionData.FromString,
                )
        self.GetSession = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcCoreService/GetSession',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
                response_deserializer=matlabs_dot_owlracer_dot_core__pb2.SessionData.FromString,
                )
        self.GetSessionIds = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcCoreService/GetSessionIds',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=matlabs_dot_owlracer_dot_core__pb2.GuidListData.FromString,
                )
        self.CreateCar = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcCoreService/CreateCar',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.CreateCarData.SerializeToString,
                response_deserializer=matlabs_dot_owlracer_dot_core__pb2.RaceCarData.FromString,
                )
        self.DestroyCar = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcCoreService/DestroyCar',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.DestroySession = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcCoreService/DestroySession',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )
        self.GetCarData = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcCoreService/GetCarData',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
                response_deserializer=matlabs_dot_owlracer_dot_core__pb2.RaceCarData.FromString,
                )
        self.Step = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcCoreService/Step',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.StepData.SerializeToString,
                response_deserializer=matlabs_dot_owlracer_dot_core__pb2.RaceCarData.FromString,
                )
        self.Reset = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcCoreService/Reset',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
                response_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                )


class GrpcCoreServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetCarIds(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CreateSession(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSession(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetSessionIds(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def CreateCar(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DestroyCar(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DestroySession(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetCarData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Step(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Reset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GrpcCoreServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetCarIds': grpc.unary_unary_rpc_method_handler(
                    servicer.GetCarIds,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.FromString,
                    response_serializer=matlabs_dot_owlracer_dot_core__pb2.GuidListData.SerializeToString,
            ),
            'CreateSession': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateSession,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.CreateSessionData.FromString,
                    response_serializer=matlabs_dot_owlracer_dot_core__pb2.SessionData.SerializeToString,
            ),
            'GetSession': grpc.unary_unary_rpc_method_handler(
                    servicer.GetSession,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.FromString,
                    response_serializer=matlabs_dot_owlracer_dot_core__pb2.SessionData.SerializeToString,
            ),
            'GetSessionIds': grpc.unary_unary_rpc_method_handler(
                    servicer.GetSessionIds,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=matlabs_dot_owlracer_dot_core__pb2.GuidListData.SerializeToString,
            ),
            'CreateCar': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateCar,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.CreateCarData.FromString,
                    response_serializer=matlabs_dot_owlracer_dot_core__pb2.RaceCarData.SerializeToString,
            ),
            'DestroyCar': grpc.unary_unary_rpc_method_handler(
                    servicer.DestroyCar,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'DestroySession': grpc.unary_unary_rpc_method_handler(
                    servicer.DestroySession,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
            'GetCarData': grpc.unary_unary_rpc_method_handler(
                    servicer.GetCarData,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.FromString,
                    response_serializer=matlabs_dot_owlracer_dot_core__pb2.RaceCarData.SerializeToString,
            ),
            'Step': grpc.unary_unary_rpc_method_handler(
                    servicer.Step,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.StepData.FromString,
                    response_serializer=matlabs_dot_owlracer_dot_core__pb2.RaceCarData.SerializeToString,
            ),
            'Reset': grpc.unary_unary_rpc_method_handler(
                    servicer.Reset,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.GuidData.FromString,
                    response_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'matlabs.owlracer.core.GrpcCoreService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GrpcCoreService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetCarIds(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcCoreService/GetCarIds',
            matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
            matlabs_dot_owlracer_dot_core__pb2.GuidListData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CreateSession(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcCoreService/CreateSession',
            matlabs_dot_owlracer_dot_core__pb2.CreateSessionData.SerializeToString,
            matlabs_dot_owlracer_dot_core__pb2.SessionData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetSession(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcCoreService/GetSession',
            matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
            matlabs_dot_owlracer_dot_core__pb2.SessionData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetSessionIds(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcCoreService/GetSessionIds',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            matlabs_dot_owlracer_dot_core__pb2.GuidListData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def CreateCar(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcCoreService/CreateCar',
            matlabs_dot_owlracer_dot_core__pb2.CreateCarData.SerializeToString,
            matlabs_dot_owlracer_dot_core__pb2.RaceCarData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DestroyCar(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcCoreService/DestroyCar',
            matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DestroySession(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcCoreService/DestroySession',
            matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetCarData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcCoreService/GetCarData',
            matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
            matlabs_dot_owlracer_dot_core__pb2.RaceCarData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Step(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcCoreService/Step',
            matlabs_dot_owlracer_dot_core__pb2.StepData.SerializeToString,
            matlabs_dot_owlracer_dot_core__pb2.RaceCarData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Reset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcCoreService/Reset',
            matlabs_dot_owlracer_dot_core__pb2.GuidData.SerializeToString,
            google_dot_protobuf_dot_empty__pb2.Empty.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class GrpcResourceServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetBaseImages = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcResourceService/GetBaseImages',
                request_serializer=google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
                response_deserializer=matlabs_dot_owlracer_dot_core__pb2.ResourceImagesDataResponse.FromString,
                )
        self.GetTrackImage = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcResourceService/GetTrackImage',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.TrackIdData.SerializeToString,
                response_deserializer=matlabs_dot_owlracer_dot_core__pb2.TrackImageDataResponse.FromString,
                )
        self.GetTrackData = channel.unary_unary(
                '/matlabs.owlracer.core.GrpcResourceService/GetTrackData',
                request_serializer=matlabs_dot_owlracer_dot_core__pb2.TrackIdData.SerializeToString,
                response_deserializer=matlabs_dot_owlracer_dot_core__pb2.TrackData.FromString,
                )


class GrpcResourceServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetBaseImages(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTrackImage(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTrackData(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_GrpcResourceServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetBaseImages': grpc.unary_unary_rpc_method_handler(
                    servicer.GetBaseImages,
                    request_deserializer=google_dot_protobuf_dot_empty__pb2.Empty.FromString,
                    response_serializer=matlabs_dot_owlracer_dot_core__pb2.ResourceImagesDataResponse.SerializeToString,
            ),
            'GetTrackImage': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTrackImage,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.TrackIdData.FromString,
                    response_serializer=matlabs_dot_owlracer_dot_core__pb2.TrackImageDataResponse.SerializeToString,
            ),
            'GetTrackData': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTrackData,
                    request_deserializer=matlabs_dot_owlracer_dot_core__pb2.TrackIdData.FromString,
                    response_serializer=matlabs_dot_owlracer_dot_core__pb2.TrackData.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'matlabs.owlracer.core.GrpcResourceService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class GrpcResourceService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetBaseImages(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcResourceService/GetBaseImages',
            google_dot_protobuf_dot_empty__pb2.Empty.SerializeToString,
            matlabs_dot_owlracer_dot_core__pb2.ResourceImagesDataResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTrackImage(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcResourceService/GetTrackImage',
            matlabs_dot_owlracer_dot_core__pb2.TrackIdData.SerializeToString,
            matlabs_dot_owlracer_dot_core__pb2.TrackImageDataResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTrackData(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/matlabs.owlracer.core.GrpcResourceService/GetTrackData',
            matlabs_dot_owlracer_dot_core__pb2.TrackIdData.SerializeToString,
            matlabs_dot_owlracer_dot_core__pb2.TrackData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
