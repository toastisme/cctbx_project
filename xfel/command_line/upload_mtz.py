from __future__ import absolute_import, division, print_function
# LIBTBX_SET_DISPATCHER_NAME cctbx.xfel.upload_mtz

from libtbx.phil import parse
from dials.util import Sorry
import os, sys
import re


help_message = """

Upload an .mtz file and merging log to a shared Google Drive folder.

"""

phil_str = """
drive {
  credential_file = None
    .type = path
    .help = Credential file (json format) for a Google Cloud service account
  shared_folder_id = None
    .type = str
    .help = Id string of the destination folder. If the folder url is \
https://drive.google.com/drive/u/0/folders/1NlJkfL6CMd1NZIl6Duy23i4G1RM9cNH- , \
then the id is 1NlJkfL6CMd1NZIl6Duy23i4G1RM9cNH- .
}
input {
  mtz_file = None
    .type = path
    .help = Location of the mtz file to upload
  log_file = None
    .type = path
    .help = Location of the log file to upload. If None, guess from mtz name.
  version = None
    .type = int
    .help = Dataset version number. If None, guess from mtz name.
}
"""
phil_scope = parse(phil_str)

def _get_root_and_version(mtz_fname):
  """
  find and return the dataset name and version string from an mtz filename
  """
  regex = re.compile(r'(.*)_(v\d{3})_all.mtz$')
  hit = regex.search(mtz_fname)
  assert hit is not None
  assert len(hit.groups()) == 2
  return hit.groups()

def _get_log_fname(mtz_fname):
  """
  convert an mtz filename to the corresponding main log filename
  """
  regex = re.compile(r'(.*)_all.mtz$')
  hit = regex.search(mtz_fname)
  assert hit is not None
  assert len(hit.groups()) == 1
  return hit.groups()[0] + '_main.log'

class pydrive2_interface:
  """
  Wrapper for uploading versioned mtzs and logs using Pydrive2. Constructed from
  a service account credentials file and the Google Drive id of the top-level
  destination folder.
  """

  def __init__(self, cred_file, folder_id):
    try:
      from pydrive2.auth import ServiceAccountCredentials, GoogleAuth
      from pydrive2.drive import GoogleDrive
    except ImportError:
      raise Sorry("Pydrive2 not found. Try:\n$ conda install pydrive2 -c conda-forge")
    gauth = GoogleAuth()
    scope = ['https://www.googleapis.com/auth/drive']
    gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
        cred_file, scope
    )
    self.drive = GoogleDrive(gauth)
    self.top_folder_id = folder_id

  def _fetch_or_create_folder(self, fname, parent_id):
    query = {
        "q": "'{}' in parents and title='{}'".format(parent_id, fname),
        "supportsTeamDrives": "true",
        "includeItemsFromAllDrives": "true",
        "corpora": "allDrives"
    }
    hits = self.drive.ListFile(query).GetList()
    if hits:
      assert len(hits)==1
      return hits[0]['id']
    else:
      query = {
        "title": fname,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [{"kind": "drive#fileLink", "id": parent_id}]
      }
      f = self.drive.CreateFile(query)
      f.Upload()
      return f['id']

  def _upload_detail(self, file_path, parent_id):
    title = os.path.split(file_path)[1]
    query = {
        "title": title,
        "parents": [{"kind": "drive#fileLink", "id": parent_id}]
    }
    f = self.drive.CreateFile(query)
    f.SetContentFile(file_path)
    f.Upload()


  def upload(self, folder_list, files):
    """
    Upload from the given file paths to a folder defined by the hierarchy in
    folder_list. So if `folders` is ['a', 'b'] and `files` is [f1, f2], then
    inside the folder defined by self.folder_id, we create nested folder a/b/
    and upload f1 and f2 to that folder.
    """
    current_folder_id = self.top_folder_id
    for fname in folder_list:
      current_folder_id = self._fetch_or_create_folder(fname, current_folder_id)
    for file in files:
      self._upload_detail(file, current_folder_id)

def run(args):

  user_phil = []
  if '--help' in args or '-h' in args:
    print(help_message)
    phil_scope.show()
    return

  for arg in args:
    try:
      user_phil.append(parse(arg))
    except Exception as e:
      raise Sorry("Unrecognized argument %s"%arg)
  params = phil_scope.fetch(sources=user_phil).extract()
  run_with_preparsed(params)



def run_with_preparsed(params):
  assert params.drive.credential_file is not None
  assert params.drive.shared_folder_id is not None
  assert params.input.mtz_file is not None


  mtz_dirname, mtz_fname = os.path.split(params.input.mtz_file)
  mtz_path = params.input.mtz_file

  if params.input.version is not None:
    dataset_root = _get_root_and_version(mtz_fname)[0]
    version_str = "v{:03d}".format(params.input.version)
  else:
    dataset_root, version_str = _get_root_and_version(mtz_fname)

  if params.input.log_file is not None:
    log_path = params.input.log_file
  else:
    log_fname = _get_log_fname(mtz_fname)
    log_path = os.path.join(mtz_dirname, log_fname)

  drive = pydrive2_interface(
      params.drive.credential_file,
      params.drive.shared_folder_id
  )
  folders = [dataset_root, version_str]
  files = [mtz_path, log_path]
  drive.upload(folders, files)


if __name__=="__main__":
  run(sys.argv[1:])
