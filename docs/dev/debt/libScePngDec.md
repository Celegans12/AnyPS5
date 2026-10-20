# libScePngDec technical debt

### Unknown function info

- [scePngDecQueryMemorySize, scePngDecCreate](../../../core/libs/prx/libScePngDec/Export.cpp) (libScePngDec) - the work memory is a fixed 0x20 bytes, because decoding uses host memory; the PS5 module asks for `((max_image_width + 0x47) & ~7) * (8 << attribute)` plus a constant, as read from the 3.20 module by prosper. A null parameter returns `0x80690003`; prosper returns `0x80690001` when it cannot read the parameter, which was not measured. Closes when: measured on a PS5
