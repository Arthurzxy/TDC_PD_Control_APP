# Host App Structure

Read the host application in this order:

1. `main.py`
2. `control.py`
3. `ui/main_window.py`
4. `usb_link.py`
5. `data_processing.py`
6. `protocol.py`
7. `models.py`
8. `storage.py`

Layer split:

- UI: `ui/main_window.py`
- Control: `control.py` is the public controller entry point; `app_controller.py`, `fpga_control.py`, and `usb_link.py` implement the control services behind it.
- Data processing: `data_processing.py` is the public acquisition / histogram / replay entry point.

## FT vendor files

The following files are kept in place as vendor/reference material only:

- `ftd3xx.py`
- `_ftd3xx_win32.py`
- `FTD3XXWU.dll`

The runtime host app does **not** import `ftd3xx.py` or call `FTD3XXWU.dll` through a local ctypes wrapper. The active FT601 path is `PyD3XXDevice` in `usb_link.py`, which follows the official PyD3XX demo flow: `FT_CreateDeviceInfoList`, `FT_GetDeviceInfoDetail`, `FT_Create`, `FT_GetPipeInformation`, `FT_ReadPipe`, `FT_WritePipe`, and `FT_Close`.
