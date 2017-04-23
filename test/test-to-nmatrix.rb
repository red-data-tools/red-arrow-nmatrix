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

class ToNMatrixTest < Test::Unit::TestCase
  test("UInt8") do
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
    tensor = Arrow::Tensor.new(Arrow::UInt8DataType.new,
                               Arrow::Buffer.new(data.flatten.pack("C*")),
                               shape,
                               nil,
                               nil)
    assert_equal(NMatrix.new(shape, data.flatten, dtype: :byte),
                 tensor.to_nmatrix)
  end

  test("Int8") do
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
    tensor = Arrow::Tensor.new(Arrow::Int8DataType.new,
                               Arrow::Buffer.new(data.flatten.pack("c*")),
                               shape,
                               nil,
                               nil)
    assert_equal(NMatrix.new(shape, data.flatten, dtype: :int8),
                 tensor.to_nmatrix)
  end

  test("Int16") do
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
    tensor = Arrow::Tensor.new(Arrow::Int16DataType.new,
                               Arrow::Buffer.new(data.flatten.pack("s*")),
                               shape,
                               nil,
                               nil)
    assert_equal(NMatrix.new(shape, data.flatten, dtype: :int16),
                 tensor.to_nmatrix)
  end

  test("Int32") do
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
    tensor = Arrow::Tensor.new(Arrow::Int32DataType.new,
                               Arrow::Buffer.new(data.flatten.pack("l*")),
                               shape,
                               nil,
                               nil)
    assert_equal(NMatrix.new(shape, data.flatten, dtype: :int32),
                 tensor.to_nmatrix)
  end

  test("Int64") do
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
    tensor = Arrow::Tensor.new(Arrow::Int64DataType.new,
                               Arrow::Buffer.new(data.flatten.pack("q*")),
                               shape,
                               nil,
                               nil)
    assert_equal(NMatrix.new(shape, data.flatten, dtype: :int64),
                 tensor.to_nmatrix)
  end

  test("Float") do
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
    tensor = Arrow::Tensor.new(Arrow::FloatDataType.new,
                               Arrow::Buffer.new(data.flatten.pack("f*")),
                               shape,
                               nil,
                               nil)
    assert_equal(NMatrix.new(shape, data.flatten, dtype: :float32),
                 tensor.to_nmatrix)
  end

  test("Double") do
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
    tensor = Arrow::Tensor.new(Arrow::DoubleDataType.new,
                               Arrow::Buffer.new(data.flatten.pack("d*")),
                               shape,
                               nil,
                               nil)
    assert_equal(NMatrix.new(shape, data.flatten, dtype: :float64),
                 tensor.to_nmatrix)
  end
end
