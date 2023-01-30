# @generated by generate_proto_mypy_stubs.py.  Do not edit!
import sys
from animalai.communicator_objects.arena_configuration_proto_pb2 import (
    ArenaConfigurationProto as animalai___communicator_objects___arena_configuration_proto_pb2___ArenaConfigurationProto,
)

from google.protobuf.descriptor import (
    Descriptor as google___protobuf___descriptor___Descriptor,
)

from google.protobuf.message import Message as google___protobuf___message___Message

from typing import (
    Mapping as typing___Mapping,
    MutableMapping as typing___MutableMapping,
    Optional as typing___Optional,
    Union as typing___Union,
)

from typing_extensions import Literal as typing_extensions___Literal

builtin___bool = bool
builtin___bytes = bytes
builtin___float = float
builtin___int = int
if sys.version_info < (3,):
    builtin___buffer = buffer
    builtin___unicode = unicode

class ArenasConfigurationsProto(google___protobuf___message___Message):
    DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
    class ArenasEntry(google___protobuf___message___Message):
        DESCRIPTOR: google___protobuf___descriptor___Descriptor = ...
        key = ...  # type: builtin___int
        @property
        def value(
            self,
        ) -> animalai___communicator_objects___arena_configuration_proto_pb2___ArenaConfigurationProto: ...
        def __init__(
            self,
            *,
            key: typing___Optional[builtin___int] = None,
            value: typing___Optional[
                animalai___communicator_objects___arena_configuration_proto_pb2___ArenaConfigurationProto
            ] = None,
        ) -> None: ...
        if sys.version_info >= (3,):
            @classmethod
            def FromString(
                cls, s: builtin___bytes
            ) -> ArenasConfigurationsProto.ArenasEntry: ...
        else:
            @classmethod
            def FromString(
                cls,
                s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode],
            ) -> ArenasConfigurationsProto.ArenasEntry: ...
        def MergeFrom(
            self, other_msg: google___protobuf___message___Message
        ) -> None: ...
        def CopyFrom(
            self, other_msg: google___protobuf___message___Message
        ) -> None: ...
        def HasField(
            self, field_name: typing_extensions___Literal["value", b"value"]
        ) -> builtin___bool: ...
        def ClearField(
            self,
            field_name: typing_extensions___Literal["key", b"key", "value", b"value"],
        ) -> None: ...
    seed = ...  # type: builtin___int
    @property
    def arenas(
        self,
    ) -> typing___MutableMapping[
        builtin___int,
        animalai___communicator_objects___arena_configuration_proto_pb2___ArenaConfigurationProto,
    ]: ...
    def __init__(
        self,
        *,
        arenas: typing___Optional[
            typing___Mapping[
                builtin___int,
                animalai___communicator_objects___arena_configuration_proto_pb2___ArenaConfigurationProto,
            ]
        ] = None,
        seed: typing___Optional[builtin___int] = None,
    ) -> None: ...
    if sys.version_info >= (3,):
        @classmethod
        def FromString(cls, s: builtin___bytes) -> ArenasConfigurationsProto: ...
    else:
        @classmethod
        def FromString(
            cls, s: typing___Union[builtin___bytes, builtin___buffer, builtin___unicode]
        ) -> ArenasConfigurationsProto: ...
    def MergeFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def CopyFrom(self, other_msg: google___protobuf___message___Message) -> None: ...
    def ClearField(
        self,
        field_name: typing_extensions___Literal["arenas", b"arenas", "seed", b"seed"],
    ) -> None: ...
