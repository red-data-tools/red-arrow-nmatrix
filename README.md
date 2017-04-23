# README

## Name

Red Arrow NMatrix

## Description

Red Arrow NMatrix is a library that provides converters between Apache Arrow's tensor data (`Arrow::Tensor`) and NMatrix's matrix data (`NMatrix`).

Red Arrow NMatrix adds `Arrow::Tensor#to_nmatrix` for Apache Arrow to NMatrix conversion. Red Arrow NMatrix adds `NMatrix#to_arrow` for NMatrix to Apache Arrow conversion.

## Install

```text
% gem install red-arrow-nmatrix
```

## Usage

```ruby
require "arrow-nmatrix"

tensor.to_nmatrix # -> An object of NMatrix

matrix.to_arrow   # -> An object of Arrow::Tensor
```

## Dependencies

* [Red Arrow](https://github.com/red-data-tools/red-arrow)

* [NMatrix](https://github.com/SciRuby/nmatrix)

## Authors

* Kouhei Sutou \<kou@clear-code.com\>

## License

Apache License 2.0. See doc/text/apache-2.0.txt for details.

(Kouhei Sutou has a right to change the license including contributed
patches.)
