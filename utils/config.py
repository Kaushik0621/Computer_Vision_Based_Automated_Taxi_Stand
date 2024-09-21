from dotenv import load_dotenv
import os


env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
print(env_path)
load_dotenv(dotenv_path=env_path)

SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = os.getenv('SMTP_PORT')
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
TO_EMAIL = os.getenv('TO_EMAIL').split(',')
json_file_path = os.getenv('JSON_FILE_PATH')
