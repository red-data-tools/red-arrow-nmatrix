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

require "mkmf-gnome2"

required_pkg_config_package(["arrow-glib", 3])

[
  ["glib2", "ext/glib2"],
  ["nmatrix", "ext/nmatrix"],
].each do |name, source_dir|
  spec = find_gem_spec(name)
  source_dir = File.join(spec.full_gem_path, source_dir)
  build_dir = source_dir
  add_depend_package_path(name, source_dir, build_dir)
end

have_func("rb_array_const_ptr", "ruby.h")

create_makefile("arrow_nmatrix")
