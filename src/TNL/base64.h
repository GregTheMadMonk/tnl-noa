/***************************************************************************
                          base64.h  -  description
                             -------------------
    begin                : Mar 20, 2020
    copyright            : (C) 2020 by Tomas Oberhuber et al.
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <cmath>    // std::ceil

namespace TNL {

// The functions in the base64 namespace are taken from the libb64 project, see
// http://sourceforge.net/projects/libb64
//
// libb64 has been placed in the public domain
namespace base64 {

// encoding

typedef enum
{
    step_A,
    step_B,
    step_C
} base64_encodestep;

typedef struct
{
    base64_encodestep step;
    char              result;
} base64_encodestate;

inline void
base64_init_encodestate(base64_encodestate *state_in)
{
    state_in->step   = step_A;
    state_in->result = 0;
}

inline char
base64_encode_value(char value_in)
{
    static const char *encoding = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (value_in > 63)
        return '=';
    return encoding[(int)value_in];
}

inline std::ptrdiff_t
base64_encode_block(const char *        plaintext_in,
                    std::size_t         length_in,
                    char *              code_out,
                    base64_encodestate *state_in)
{
    const char *      plainchar    = plaintext_in;
    const char *const plaintextend = plaintext_in + length_in;
    char *            codechar     = code_out;
    char              result;

    result = state_in->result;

    switch (state_in->step)
    {
        while (true)
        {
            case step_A:
            {
                if (plainchar == plaintextend)
                {
                    state_in->result = result;
                    state_in->step   = step_A;
                    return codechar - code_out;
                }
                const char fragment = *plainchar++;
                result              = (fragment & 0x0fc) >> 2;
                *codechar++         = base64_encode_value(result);
                result              = (fragment & 0x003) << 4;
                // intended fallthrough
            }
            case step_B:
            {
                if (plainchar == plaintextend)
                {
                    state_in->result = result;
                    state_in->step   = step_B;
                    return codechar - code_out;
                }
                const char fragment = *plainchar++;
                result |= (fragment & 0x0f0) >> 4;
                *codechar++ = base64_encode_value(result);
                result      = (fragment & 0x00f) << 2;
                // intended fallthrough
            }
            case step_C:
            {
                if (plainchar == plaintextend)
                {
                    state_in->result = result;
                    state_in->step   = step_C;
                    return codechar - code_out;
                }
                const char fragment = *plainchar++;
                result |= (fragment & 0x0c0) >> 6;
                *codechar++ = base64_encode_value(result);
                result      = (fragment & 0x03f) >> 0;
                *codechar++ = base64_encode_value(result);
            }
        }
    }
    /* control should not reach here */
    return codechar - code_out;
}

inline std::ptrdiff_t
base64_encode_blockend(char *code_out, base64_encodestate *state_in)
{
    char *codechar = code_out;

    switch (state_in->step)
    {
        case step_B:
            *codechar++ = base64_encode_value(state_in->result);
            *codechar++ = '=';
            *codechar++ = '=';
            break;
        case step_C:
            *codechar++ = base64_encode_value(state_in->result);
            *codechar++ = '=';
            break;
        case step_A:
            break;
    }
    *codechar++ = '\0';

    return codechar - code_out;
}


// decoding

typedef enum
{
    step_a, step_b, step_c, step_d
} base64_decodestep;

typedef struct
{
    base64_decodestep step;
    char plainchar;
} base64_decodestate;

inline void
base64_init_decodestate(base64_decodestate* state_in)
{
    state_in->step = step_a;
    state_in->plainchar = 0;
}

inline int
base64_decode_value(char value_in)
{
    static const char decoding[] = {62,-1,-1,-1,63,52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-2,-1,-1,-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,-1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51};
    static const char decoding_size = sizeof(decoding);
    value_in -= 43;
    if (value_in < 0 || value_in >= decoding_size) return -1;
    return decoding[(int)value_in];
}

inline std::ptrdiff_t
base64_decode_block(const char* code_in, const std::size_t length_in, char* plaintext_out, base64_decodestate* state_in)
{
    const char* codechar = code_in;
    char* plainchar = plaintext_out;
    char fragment;

    *plainchar = state_in->plainchar;

    switch (state_in->step)
    {
        while (1)
        {
    case step_a:
            do {
                if (codechar == code_in+length_in)
                {
                    state_in->step = step_a;
                    state_in->plainchar = *plainchar;
                    return plainchar - plaintext_out;
                }
                fragment = (char)base64_decode_value(*codechar++);
            } while (fragment < 0);
            *plainchar    = (fragment & 0x03f) << 2;
    case step_b:
            do {
                if (codechar == code_in+length_in)
                {
                    state_in->step = step_b;
                    state_in->plainchar = *plainchar;
                    return plainchar - plaintext_out;
                }
                fragment = (char)base64_decode_value(*codechar++);
            } while (fragment < 0);
            *plainchar++ |= (fragment & 0x030) >> 4;
            *plainchar    = (fragment & 0x00f) << 4;
    case step_c:
            do {
                if (codechar == code_in+length_in)
                {
                    state_in->step = step_c;
                    state_in->plainchar = *plainchar;
                    return plainchar - plaintext_out;
                }
                fragment = (char)base64_decode_value(*codechar++);
            } while (fragment < 0);
            *plainchar++ |= (fragment & 0x03c) >> 2;
            *plainchar    = (fragment & 0x003) << 6;
    case step_d:
            do {
                if (codechar == code_in+length_in)
                {
                    state_in->step = step_d;
                    state_in->plainchar = *plainchar;
                    return plainchar - plaintext_out;
                }
                fragment = (char)base64_decode_value(*codechar++);
            } while (fragment < 0);
            *plainchar++   |= (fragment & 0x03f);
        }
    }
    /* control should not reach here */
    return plainchar - plaintext_out;
}

} // namespace base64


/**
 * Do a base64 encoding of the given data.
 *
 * The function returns a unique_ptr to the encoded data.
 */
inline std::unique_ptr<char[]>
encode_block(const char* data, const std::size_t data_size)
{
    base64::base64_encodestate state;
    base64::base64_init_encodestate(&state);

    std::unique_ptr<char[]> encoded_data{new char[2 * data_size + 1]};

    const std::size_t encoded_length_data = base64::base64_encode_block(data, data_size, encoded_data.get(), &state);
    base64::base64_encode_blockend(encoded_data.get() + encoded_length_data, &state);

    return encoded_data;
}


/**
 * Do a base64 decoding of the given data.
 *
 * The function returns a pair of the decoded data length and a unique_ptr to
 * the decoded data.
 */
inline std::pair<std::size_t, std::unique_ptr<char[]>>
decode_block(const char* data, const std::size_t data_size)
{
    base64::base64_decodestate state;
    base64::base64_init_decodestate(&state);

    std::unique_ptr<char[]> decoded_data{new char[data_size + 1]};

    const std::size_t decoded_length_data = base64::base64_decode_block(data, data_size, decoded_data.get(), &state);
    decoded_data[decoded_length_data] = '\0';

    return {decoded_length_data, std::move(decoded_data)};
}


/**
 * Write a base64-encoded block of data into the given stream.
 *
 * The encoded data is prepended with a short header, which is the base64-encoded
 * byte length of the data. The type of the byte length value is `HeaderType`.
 */
template <typename HeaderType = std::uint64_t, typename T>
void write_encoded_block(const T* data, const std::size_t data_length, std::ostream& output_stream)
{
   const HeaderType size = data_length * sizeof(T);
   std::unique_ptr<char[]> encoded_size = encode_block(reinterpret_cast<const char*>(&size), sizeof(HeaderType));
   output_stream << encoded_size.get();
   std::unique_ptr<char[]> encoded_data = encode_block(reinterpret_cast<const char*>(data), size);
   output_stream << encoded_data.get();
}


/**
 * Get the length of base64-encoded block for given data byte length.
 */
inline std::size_t
get_encoded_length(const std::size_t byte_length)
{
    int encoded = std::ceil(byte_length * (4.0 / 3.0));
    // base64 uses padding to a multiple of 4
    if (encoded % 4 == 0)
        return encoded;
    return encoded + 4 - (encoded % 4);
}

} // namespace TNL
