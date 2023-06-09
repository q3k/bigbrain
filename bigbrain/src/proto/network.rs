// This file is generated by rust-protobuf 2.22.1. Do not edit
// @generated

// https://github.com/rust-lang/rust-clippy/issues/702
#![allow(unknown_lints)]
#![allow(clippy::all)]

#![allow(unused_attributes)]
#![cfg_attr(rustfmt, rustfmt::skip)]

#![allow(box_pointers)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(trivial_casts)]
#![allow(unused_imports)]
#![allow(unused_results)]
//! Generated file from `proto/network.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
// const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_2_22_1;

#[derive(PartialEq,Clone,Default)]
pub struct InputLayer {
    // message fields
    pub size: u32,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a InputLayer {
    fn default() -> &'a InputLayer {
        <InputLayer as ::protobuf::Message>::default_instance()
    }
}

impl InputLayer {
    pub fn new() -> InputLayer {
        ::std::default::Default::default()
    }

    // uint32 size = 1;


    pub fn get_size(&self) -> u32 {
        self.size
    }
    pub fn clear_size(&mut self) {
        self.size = 0;
    }

    // Param is passed by value, moved
    pub fn set_size(&mut self, v: u32) {
        self.size = v;
    }
}

impl ::protobuf::Message for InputLayer {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_uint32()?;
                    self.size = tmp;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if self.size != 0 {
            my_size += ::protobuf::rt::value_size(1, self.size, ::protobuf::wire_format::WireTypeVarint);
        }
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if self.size != 0 {
            os.write_uint32(1, self.size)?;
        }
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn (::std::any::Any) {
        self as &dyn (::std::any::Any)
    }
    fn as_any_mut(&mut self) -> &mut dyn (::std::any::Any) {
        self as &mut dyn (::std::any::Any)
    }
    fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn (::std::any::Any)> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        Self::descriptor_static()
    }

    fn new() -> InputLayer {
        InputLayer::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeUint32>(
                "size",
                |m: &InputLayer| { &m.size },
                |m: &mut InputLayer| { &mut m.size },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<InputLayer>(
                "InputLayer",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static InputLayer {
        static instance: ::protobuf::rt::LazyV2<InputLayer> = ::protobuf::rt::LazyV2::INIT;
        instance.get(InputLayer::new)
    }
}

impl ::protobuf::Clear for InputLayer {
    fn clear(&mut self) {
        self.size = 0;
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for InputLayer {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for InputLayer {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct InnerLayer {
    // message fields
    pub size: u32,
    pub weights: ::std::vec::Vec<f32>,
    pub biases: ::std::vec::Vec<f32>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a InnerLayer {
    fn default() -> &'a InnerLayer {
        <InnerLayer as ::protobuf::Message>::default_instance()
    }
}

impl InnerLayer {
    pub fn new() -> InnerLayer {
        ::std::default::Default::default()
    }

    // uint32 size = 1;


    pub fn get_size(&self) -> u32 {
        self.size
    }
    pub fn clear_size(&mut self) {
        self.size = 0;
    }

    // Param is passed by value, moved
    pub fn set_size(&mut self, v: u32) {
        self.size = v;
    }

    // repeated float weights = 2;


    pub fn get_weights(&self) -> &[f32] {
        &self.weights
    }
    pub fn clear_weights(&mut self) {
        self.weights.clear();
    }

    // Param is passed by value, moved
    pub fn set_weights(&mut self, v: ::std::vec::Vec<f32>) {
        self.weights = v;
    }

    // Mutable pointer to the field.
    pub fn mut_weights(&mut self) -> &mut ::std::vec::Vec<f32> {
        &mut self.weights
    }

    // Take field
    pub fn take_weights(&mut self) -> ::std::vec::Vec<f32> {
        ::std::mem::replace(&mut self.weights, ::std::vec::Vec::new())
    }

    // repeated float biases = 3;


    pub fn get_biases(&self) -> &[f32] {
        &self.biases
    }
    pub fn clear_biases(&mut self) {
        self.biases.clear();
    }

    // Param is passed by value, moved
    pub fn set_biases(&mut self, v: ::std::vec::Vec<f32>) {
        self.biases = v;
    }

    // Mutable pointer to the field.
    pub fn mut_biases(&mut self) -> &mut ::std::vec::Vec<f32> {
        &mut self.biases
    }

    // Take field
    pub fn take_biases(&mut self) -> ::std::vec::Vec<f32> {
        ::std::mem::replace(&mut self.biases, ::std::vec::Vec::new())
    }
}

impl ::protobuf::Message for InnerLayer {
    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    if wire_type != ::protobuf::wire_format::WireTypeVarint {
                        return ::std::result::Result::Err(::protobuf::rt::unexpected_wire_type(wire_type));
                    }
                    let tmp = is.read_uint32()?;
                    self.size = tmp;
                },
                2 => {
                    ::protobuf::rt::read_repeated_float_into(wire_type, is, &mut self.weights)?;
                },
                3 => {
                    ::protobuf::rt::read_repeated_float_into(wire_type, is, &mut self.biases)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if self.size != 0 {
            my_size += ::protobuf::rt::value_size(1, self.size, ::protobuf::wire_format::WireTypeVarint);
        }
        my_size += 5 * self.weights.len() as u32;
        my_size += 5 * self.biases.len() as u32;
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if self.size != 0 {
            os.write_uint32(1, self.size)?;
        }
        for v in &self.weights {
            os.write_float(2, *v)?;
        };
        for v in &self.biases {
            os.write_float(3, *v)?;
        };
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn (::std::any::Any) {
        self as &dyn (::std::any::Any)
    }
    fn as_any_mut(&mut self) -> &mut dyn (::std::any::Any) {
        self as &mut dyn (::std::any::Any)
    }
    fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn (::std::any::Any)> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        Self::descriptor_static()
    }

    fn new() -> InnerLayer {
        InnerLayer::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_simple_field_accessor::<_, ::protobuf::types::ProtobufTypeUint32>(
                "size",
                |m: &InnerLayer| { &m.size },
                |m: &mut InnerLayer| { &mut m.size },
            ));
            fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeFloat>(
                "weights",
                |m: &InnerLayer| { &m.weights },
                |m: &mut InnerLayer| { &mut m.weights },
            ));
            fields.push(::protobuf::reflect::accessor::make_vec_accessor::<_, ::protobuf::types::ProtobufTypeFloat>(
                "biases",
                |m: &InnerLayer| { &m.biases },
                |m: &mut InnerLayer| { &mut m.biases },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<InnerLayer>(
                "InnerLayer",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static InnerLayer {
        static instance: ::protobuf::rt::LazyV2<InnerLayer> = ::protobuf::rt::LazyV2::INIT;
        instance.get(InnerLayer::new)
    }
}

impl ::protobuf::Clear for InnerLayer {
    fn clear(&mut self) {
        self.size = 0;
        self.weights.clear();
        self.biases.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for InnerLayer {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for InnerLayer {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

#[derive(PartialEq,Clone,Default)]
pub struct Network {
    // message fields
    pub input: ::protobuf::SingularPtrField<InputLayer>,
    pub inner: ::protobuf::RepeatedField<InnerLayer>,
    // special fields
    pub unknown_fields: ::protobuf::UnknownFields,
    pub cached_size: ::protobuf::CachedSize,
}

impl<'a> ::std::default::Default for &'a Network {
    fn default() -> &'a Network {
        <Network as ::protobuf::Message>::default_instance()
    }
}

impl Network {
    pub fn new() -> Network {
        ::std::default::Default::default()
    }

    // .neural.network.InputLayer input = 1;


    pub fn get_input(&self) -> &InputLayer {
        self.input.as_ref().unwrap_or_else(|| <InputLayer as ::protobuf::Message>::default_instance())
    }
    pub fn clear_input(&mut self) {
        self.input.clear();
    }

    pub fn has_input(&self) -> bool {
        self.input.is_some()
    }

    // Param is passed by value, moved
    pub fn set_input(&mut self, v: InputLayer) {
        self.input = ::protobuf::SingularPtrField::some(v);
    }

    // Mutable pointer to the field.
    // If field is not initialized, it is initialized with default value first.
    pub fn mut_input(&mut self) -> &mut InputLayer {
        if self.input.is_none() {
            self.input.set_default();
        }
        self.input.as_mut().unwrap()
    }

    // Take field
    pub fn take_input(&mut self) -> InputLayer {
        self.input.take().unwrap_or_else(|| InputLayer::new())
    }

    // repeated .neural.network.InnerLayer inner = 2;


    pub fn get_inner(&self) -> &[InnerLayer] {
        &self.inner
    }
    pub fn clear_inner(&mut self) {
        self.inner.clear();
    }

    // Param is passed by value, moved
    pub fn set_inner(&mut self, v: ::protobuf::RepeatedField<InnerLayer>) {
        self.inner = v;
    }

    // Mutable pointer to the field.
    pub fn mut_inner(&mut self) -> &mut ::protobuf::RepeatedField<InnerLayer> {
        &mut self.inner
    }

    // Take field
    pub fn take_inner(&mut self) -> ::protobuf::RepeatedField<InnerLayer> {
        ::std::mem::replace(&mut self.inner, ::protobuf::RepeatedField::new())
    }
}

impl ::protobuf::Message for Network {
    fn is_initialized(&self) -> bool {
        for v in &self.input {
            if !v.is_initialized() {
                return false;
            }
        };
        for v in &self.inner {
            if !v.is_initialized() {
                return false;
            }
        };
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        while !is.eof()? {
            let (field_number, wire_type) = is.read_tag_unpack()?;
            match field_number {
                1 => {
                    ::protobuf::rt::read_singular_message_into(wire_type, is, &mut self.input)?;
                },
                2 => {
                    ::protobuf::rt::read_repeated_message_into(wire_type, is, &mut self.inner)?;
                },
                _ => {
                    ::protobuf::rt::read_unknown_or_skip_group(field_number, wire_type, is, self.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u32 {
        let mut my_size = 0;
        if let Some(ref v) = self.input.as_ref() {
            let len = v.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        }
        for value in &self.inner {
            let len = value.compute_size();
            my_size += 1 + ::protobuf::rt::compute_raw_varint32_size(len) + len;
        };
        my_size += ::protobuf::rt::unknown_fields_size(self.get_unknown_fields());
        self.cached_size.set(my_size);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::ProtobufResult<()> {
        if let Some(ref v) = self.input.as_ref() {
            os.write_tag(1, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        }
        for v in &self.inner {
            os.write_tag(2, ::protobuf::wire_format::WireTypeLengthDelimited)?;
            os.write_raw_varint32(v.get_cached_size())?;
            v.write_to_with_cached_sizes(os)?;
        };
        os.write_unknown_fields(self.get_unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn get_cached_size(&self) -> u32 {
        self.cached_size.get()
    }

    fn get_unknown_fields(&self) -> &::protobuf::UnknownFields {
        &self.unknown_fields
    }

    fn mut_unknown_fields(&mut self) -> &mut ::protobuf::UnknownFields {
        &mut self.unknown_fields
    }

    fn as_any(&self) -> &dyn (::std::any::Any) {
        self as &dyn (::std::any::Any)
    }
    fn as_any_mut(&mut self) -> &mut dyn (::std::any::Any) {
        self as &mut dyn (::std::any::Any)
    }
    fn into_any(self: ::std::boxed::Box<Self>) -> ::std::boxed::Box<dyn (::std::any::Any)> {
        self
    }

    fn descriptor(&self) -> &'static ::protobuf::reflect::MessageDescriptor {
        Self::descriptor_static()
    }

    fn new() -> Network {
        Network::new()
    }

    fn descriptor_static() -> &'static ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::LazyV2<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::LazyV2::INIT;
        descriptor.get(|| {
            let mut fields = ::std::vec::Vec::new();
            fields.push(::protobuf::reflect::accessor::make_singular_ptr_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<InputLayer>>(
                "input",
                |m: &Network| { &m.input },
                |m: &mut Network| { &mut m.input },
            ));
            fields.push(::protobuf::reflect::accessor::make_repeated_field_accessor::<_, ::protobuf::types::ProtobufTypeMessage<InnerLayer>>(
                "inner",
                |m: &Network| { &m.inner },
                |m: &mut Network| { &mut m.inner },
            ));
            ::protobuf::reflect::MessageDescriptor::new_pb_name::<Network>(
                "Network",
                fields,
                file_descriptor_proto()
            )
        })
    }

    fn default_instance() -> &'static Network {
        static instance: ::protobuf::rt::LazyV2<Network> = ::protobuf::rt::LazyV2::INIT;
        instance.get(Network::new)
    }
}

impl ::protobuf::Clear for Network {
    fn clear(&mut self) {
        self.input.clear();
        self.inner.clear();
        self.unknown_fields.clear();
    }
}

impl ::std::fmt::Debug for Network {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for Network {
    fn as_ref(&self) -> ::protobuf::reflect::ReflectValueRef {
        ::protobuf::reflect::ReflectValueRef::Message(self)
    }
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n\x13proto/network.proto\x12\x0eneural.network\"\x20\n\nInputLayer\x12\
    \x12\n\x04size\x18\x01\x20\x01(\rR\x04size\"R\n\nInnerLayer\x12\x12\n\
    \x04size\x18\x01\x20\x01(\rR\x04size\x12\x18\n\x07weights\x18\x02\x20\
    \x03(\x02R\x07weights\x12\x16\n\x06biases\x18\x03\x20\x03(\x02R\x06biase\
    s\"m\n\x07Network\x120\n\x05input\x18\x01\x20\x01(\x0b2\x1a.neural.netwo\
    rk.InputLayerR\x05input\x120\n\x05inner\x18\x02\x20\x03(\x0b2\x1a.neural\
    .network.InnerLayerR\x05innerJ\xd8\x03\n\x06\x12\x04\0\0\x10\x01\n\x08\n\
    \x01\x0c\x12\x03\0\0\x12\n\x08\n\x01\x02\x12\x03\x01\0\x17\n\n\n\x02\x04\
    \0\x12\x04\x03\0\x05\x01\n\n\n\x03\x04\0\x01\x12\x03\x03\x08\x12\n\x0b\n\
    \x04\x04\0\x02\0\x12\x03\x04\x04\x14\n\x0c\n\x05\x04\0\x02\0\x05\x12\x03\
    \x04\x04\n\n\x0c\n\x05\x04\0\x02\0\x01\x12\x03\x04\x0b\x0f\n\x0c\n\x05\
    \x04\0\x02\0\x03\x12\x03\x04\x12\x13\n\n\n\x02\x04\x01\x12\x04\x07\0\x0b\
    \x01\n\n\n\x03\x04\x01\x01\x12\x03\x07\x08\x12\n\x0b\n\x04\x04\x01\x02\0\
    \x12\x03\x08\x04\x14\n\x0c\n\x05\x04\x01\x02\0\x05\x12\x03\x08\x04\n\n\
    \x0c\n\x05\x04\x01\x02\0\x01\x12\x03\x08\x0b\x0f\n\x0c\n\x05\x04\x01\x02\
    \0\x03\x12\x03\x08\x12\x13\n\x0b\n\x04\x04\x01\x02\x01\x12\x03\t\x04\x1f\
    \n\x0c\n\x05\x04\x01\x02\x01\x04\x12\x03\t\x04\x0c\n\x0c\n\x05\x04\x01\
    \x02\x01\x05\x12\x03\t\r\x12\n\x0c\n\x05\x04\x01\x02\x01\x01\x12\x03\t\
    \x13\x1a\n\x0c\n\x05\x04\x01\x02\x01\x03\x12\x03\t\x1d\x1e\n\x0b\n\x04\
    \x04\x01\x02\x02\x12\x03\n\x04\x1e\n\x0c\n\x05\x04\x01\x02\x02\x04\x12\
    \x03\n\x04\x0c\n\x0c\n\x05\x04\x01\x02\x02\x05\x12\x03\n\r\x12\n\x0c\n\
    \x05\x04\x01\x02\x02\x01\x12\x03\n\x13\x19\n\x0c\n\x05\x04\x01\x02\x02\
    \x03\x12\x03\n\x1c\x1d\n\n\n\x02\x04\x02\x12\x04\r\0\x10\x01\n\n\n\x03\
    \x04\x02\x01\x12\x03\r\x08\x0f\n\x0b\n\x04\x04\x02\x02\0\x12\x03\x0e\x04\
    \x19\n\x0c\n\x05\x04\x02\x02\0\x06\x12\x03\x0e\x04\x0e\n\x0c\n\x05\x04\
    \x02\x02\0\x01\x12\x03\x0e\x0f\x14\n\x0c\n\x05\x04\x02\x02\0\x03\x12\x03\
    \x0e\x17\x18\n\x0b\n\x04\x04\x02\x02\x01\x12\x03\x0f\x04\"\n\x0c\n\x05\
    \x04\x02\x02\x01\x04\x12\x03\x0f\x04\x0c\n\x0c\n\x05\x04\x02\x02\x01\x06\
    \x12\x03\x0f\r\x17\n\x0c\n\x05\x04\x02\x02\x01\x01\x12\x03\x0f\x18\x1d\n\
    \x0c\n\x05\x04\x02\x02\x01\x03\x12\x03\x0f\x20!b\x06proto3\
";

static file_descriptor_proto_lazy: ::protobuf::rt::LazyV2<::protobuf::descriptor::FileDescriptorProto> = ::protobuf::rt::LazyV2::INIT;

fn parse_descriptor_proto() -> ::protobuf::descriptor::FileDescriptorProto {
    ::protobuf::Message::parse_from_bytes(file_descriptor_proto_data).unwrap()
}

pub fn file_descriptor_proto() -> &'static ::protobuf::descriptor::FileDescriptorProto {
    file_descriptor_proto_lazy.get(|| {
        parse_descriptor_proto()
    })
}
