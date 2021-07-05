from __future__ import absolute_import, division, print_function

from wxtbx import phil_controls
from wxtbx.phil_controls.text_base import TextCtrlValidator
from libtbx.utils import Abort
import wx

class SpaceGroupCtrl(wx.ComboBox, phil_controls.PhilCtrl):
  def __init__(self, *args, **kwds):
    kwds = dict(kwds)
    saved_value = None
    if (kwds.get('value', '') != ""):
      saved_value = kwds['value']
      kwds['value'] = ""
    super(SpaceGroupCtrl, self).__init__(*args, **kwds)
    style = self.GetWindowStyle()
    self.SetWindowStyle(style | wx.TE_PROCESS_ENTER)
    self.SetValidator(SpaceGroupValidator())
    self.Bind(wx.EVT_TEXT_ENTER, lambda evt: self.Validate(), self)
    self.Bind(wx.EVT_KILL_FOCUS , lambda evt: self.OnKillFocus(evt), self)
    if (saved_value is not None):
      self.SetSpaceGroup(saved_value)
    self.SetToolTip(wx.ToolTip("You may use any valid space group symbol, "+
      "e.g. 'P212121', 'P 21 21 21', 'P2(1)2(1)2(1)', or '19'; these will be "+
      "converted internally to a standard format.  You can trigger this "+
      "conversion by entering a symbol and pressing 'Enter'."))
    if (wx.Platform == '__WXMAC__') and (wx.PlatformInfo[4] == 'wxOSX-cocoa'):
      self.SetMinSize(self.GetSize())

  def SetSpaceGroup(self, sg):
    from cctbx import sgtbx
    assert (type(sg).__name__ in ["NoneType", "str", "space_group", "unicode",
      "space_group_info"])
    items = self.GetItems()
    if (isinstance(sg, sgtbx.space_group)):
      sg = str(sgtbx.space_group_info(group=sg))
    elif (sg is not None):
      sg = str(sg)
    else :
      sg = ""
    if (not sg in items):
      self.Append(sg)
    self.SetStringSelection(sg)

  def SetValue(self, sg):
    self.SetSpaceGroup(sg)

  def GetPhilValue(self):
    self.Validate()
    sg = str(self.GetValue())
    if (sg == ""):
      return self.ReturnNoneIfOptional()
    from cctbx import sgtbx
    return sgtbx.space_group_info(symbol=sg)

  def GetStringValue(self):
    return str(self.GetPhilValue())

  def OnKillFocus(self, evt):
    if hasattr(self.GetParent(), 'get_philhandler_and_object'):
      # validate on Phaser MR whenever we loose focus
      philhandler, philobject = self.GetParent().get_philhandler_and_object()
      philvalue = self.GetPhilValue()
      from wxGUI2.Programs import Extensions
      Extensions.check_point_group(philhandler, philobject, philvalue)
    evt.Skip()

  def Validate(self):
    # XXX why doesn't self.Validate() work?
    if self.GetValidator().Validate(self.GetParent()):
      return True
    else :
      raise Abort()

  if (wx.Platform == "__WXMAC__") and (wx.PlatformInfo[4] != 'wxOSX-cocoa') \
    and wx.version().startswith('3'):
    def SetBackgroundColour(self, color):
      self.GetChildren()[0].SetBackgroundColour(color)

class SpaceGroupValidator(TextCtrlValidator):
  def CheckFormat(self, value):
    if len(value) == 0 :
      return ""
    from cctbx import sgtbx
    sg = str(sgtbx.space_group_info(symbol=str(value)))
    return sg

if (__name__ == "__main__"):
  app = wx.App(0)
  frame = wx.Frame(None, -1, "Space group test")
  panel = wx.Panel(frame, -1, size=(600,400))
  txt1 = wx.StaticText(panel, -1, "Space group:", pos=(100,180))
  sg_ctrl = SpaceGroupCtrl(panel, -1, pos=(200,180),
    name="Space group")
  btn = wx.Button(panel, -1, "Process input", pos=(400, 360))
  def OnOkay(evt):
    sg = sg_ctrl.GetPhilValue()
    print(type(sg).__name__, str(sg))
  frame.Bind(wx.EVT_BUTTON, OnOkay, btn)
  frame.Fit()
  frame.Show()
  app.MainLoop()
