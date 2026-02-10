import ee
from pydrive2.auth import GoogleAuth
import argparse

def auth_ee():
	if ee.Authenticate():
		print("GEE is already authenticated")

def auth_pydrive():
	gauth = GoogleAuth(settings_file="src/pydrive_settings.yaml")
	if gauth.credentials is None:
		gauth.LocalWebserverAuth()
	else:
		print("PyDrive is already authenticated")

parser = argparse.ArgumentParser()
parser.add_argument('--skip-ee', action='store_true')
parser.add_argument('--skip-pydrive', action='store_true')

if __name__ == "__main__":	  
	args = parser.parse_args()

	if not args.skip_ee:
		auth_ee()
	
	if not args.skip_pydrive:
		auth_pydrive()