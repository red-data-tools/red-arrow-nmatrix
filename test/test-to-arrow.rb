# Copyright 2017 Kouhei Sutou <kou@clear-code.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

class ToArrowTest < Test::Unit::TestCase
  test(":byte") do
    data = [
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
      [
        [9, 10, 11, 12],
        [13, 14, 15, 16],
      ],
      [
        [17, 18, 19, 20],
        [21, 22, 23, 24],
      ],
    ]
    shape = [3, 2, 4]
    nmatrix = NMatrix.new(shape, data.flatten, dtype: :byte)
    tensor = nmatrix.to_arrow
    assert_equal([
                   Arrow::Type::UINT8,
                   data.flatten,
                 ],
                 [
                   tensor.value_type,
                   tensor.buffer.data.to_s.unpack("C*"),
                 ])
  end

  test(":int8") do
    data = [
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
      [
        [9, 10, 11, 12],
        [-1, -2, -3, -4],
      ],
      [
        [-5, -6, -7, -8],
        [-9, -10, -11, -12],
      ],
    ]
    shape = [3, 2, 4]
    nmatrix = NMatrix.new(shape, data.flatten, dtype: :int8)
    tensor = nmatrix.to_arrow
    assert_equal([
                   Arrow::Type::INT8,
                   data.flatten,
                 ],
                 [
                   tensor.value_type,
                   tensor.buffer.data.to_s.unpack("c*"),
                 ])
  end

  test(":int16") do
    data = [
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
      [
        [9, 10, 11, 12],
        [-1, -2, -3, -4],
      ],
      [
        [-5, -6, -7, -8],
        [-9, -10, -11, -12],
      ],
    ]
    shape = [3, 2, 4]
    nmatrix = NMatrix.new(shape, data.flatten, dtype: :int16)
    tensor = nmatrix.to_arrow
    assert_equal([
                   Arrow::Type::INT16,
                   data.flatten,
                 ],
                 [
                   tensor.value_type,
                   tensor.buffer.data.to_s.unpack("s*"),
                 ])
  end

  test(":int32") do
    data = [
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
      [
        [9, 10, 11, 12],
        [-1, -2, -3, -4],
      ],
      [
        [-5, -6, -7, -8],
        [-9, -10, -11, -12],
      ],
    ]
    shape = [3, 2, 4]
    nmatrix = NMatrix.new(shape, data.flatten, dtype: :int32)
    tensor = nmatrix.to_arrow
    assert_equal([
                   Arrow::Type::INT32,
                   data.flatten,
                 ],
                 [
                   tensor.value_type,
                   tensor.buffer.data.to_s.unpack("l*"),
                 ])
  end

  test(":int64") do
    data = [
      [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
      ],
      [
        [9, 10, 11, 12],
        [-1, -2, -3, -4],
      ],
      [
        [-5, -6, -7, -8],
        [-9, -10, -11, -12],
      ],
    ]
    shape = [3, 2, 4]
    nmatrix = NMatrix.new(shape, data.flatten, dtype: :int64)
    tensor = nmatrix.to_arrow
    assert_equal([
                   Arrow::Type::INT64,
                   data.flatten,
                 ],
                 [
                   tensor.value_type,
                   tensor.buffer.data.to_s.unpack("q*"),
                 ])
  end

  test(":float32") do
    data = [
      [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
      ],
      [
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
      ],
      [
        [17.0, 18.0, 19.0, 20.0],
        [21.0, 22.0, 23.0, 24.0],
      ],
    ]
    shape = [3, 2, 4]
    nmatrix = NMatrix.new(shape, data.flatten, dtype: :float32)
    tensor = nmatrix.to_arrow
    assert_equal([
                   Arrow::Type::FLOAT,
                   data.flatten,
                 ],
                 [
                   tensor.value_type,
                   tensor.buffer.data.to_s.unpack("f*"),
                 ])
  end

  test(":float64") do
    data = [
      [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
      ],
      [
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
      ],
      [
        [17.0, 18.0, 19.0, 20.0],
        [21.0, 22.0, 23.0, 24.0],
      ],
    ]
    shape = [3, 2, 4]
    nmatrix = NMatrix.new(shape, data.flatten, dtype: :float64)
    tensor = nmatrix.to_arrow
    assert_equal([
                   Arrow::Type::DOUBLE,
                   data.flatten,
                 ],
                 [
                   tensor.value_type,
                   tensor.buffer.data.to_s.unpack("d*"),
                 ])
  end
end
