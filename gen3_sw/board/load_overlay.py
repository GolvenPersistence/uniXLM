"""Load the v3 Node overlay on PYNQ-ZU and set pl_clk0 (the clocking wizard's input) to 100 MHz.

    sudo env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 load_overlay.py gen3_node_v3.bit

The overlay (gen3_node_v3.bit + gen3_node_v3.hwh, gen3_fpga/node_board_v3) is downloaded through PYNQ (which also
gives XRT the device it allocates buffers on); the Node comes up held in reset. gen3rt (BoardDevice) then maps
the register window at 0xA000_0000, checks the id, sets the command ring and releases the Node.
"""
import sys
import time

from pynq import MMIO, Clocks, Overlay

bit = sys.argv[1] if len(sys.argv) > 1 else "gen3_node_v3.bit"
Overlay(bit)
Clocks.fclk0_mhz = 100.0
time.sleep(0.05)
mm = MMIO(0xA000_0000, 0x10000)
ident = mm.read(0xD * 16)
state = mm.read(0xC * 16)
print(f"overlay {bit}: fclk0 {Clocks.fclk0_mhz:.2f} MHz, Node id {ident:08x}, state {state:x}")
if ident != 0x4733_4E03:
    sys.exit("not the v3 Node")
