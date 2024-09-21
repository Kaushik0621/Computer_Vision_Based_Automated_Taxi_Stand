import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import config

# Function to send an email
def send_email(subject, body):
    # Create a MIME message
    msg = MIMEMultipart()
    msg['From'] = config.SMTP_USER
    msg['To'] = ', '.join(config.TO_EMAIL)  # Ensure this is a string
    msg['Subject'] = subject
    
    # Attach the email body in HTML format
    msg.attach(MIMEText(body, 'html'))
    
    # Establish a connection to the SMTP server
    server = smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT)
    server.starttls()  # Upgrade the connection to secure (TLS/SSL)
    
    try:
        # Log in to the SMTP server
        server.login(config.SMTP_USER, config.SMTP_PASSWORD)
        
        # Send the email
        server.sendmail(config.SMTP_USER, config.TO_EMAIL, msg.as_string())
        print(f"Email sent to {config.TO_EMAIL}")
        
    except Exception as e:
        print(f"Failed to send email. Error: {e}")
        
    finally:
        # Close the server connection
        server.quit()

# Load the JSON data from a static file
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Main function to read the JSON and send the emails
def main():
    # Load data from the JSON file
    camera_data = load_json_data(config.json_file_path)
    
    # Construct the email subject
    subject = "Camera Information Report"
    
    # Start building the email body with HTML content
    body = """
    <html>
    <body>
        <p>Here is the camera information:</p>
        <table border="1" cellpadding="5" cellspacing="0">
            <tr>
                <th>Camera</th>
                <th>Number of People</th>
                <th>Google Map Link</th>
            </tr>
    """
    
    # Loop through the JSON data and append the information to the body
    for camera, info in camera_data.items():
        location = info['location']
        number = info['number']
        lat, lon = location  # Unpack the coordinates
        google_map_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
        
        body += f"""
            <tr>
                <td>{camera}</td>
                <td>{number}</td>
                <td><a href="{google_map_link}">View on Map</a></td>
            </tr>
        """
    
    # Close the HTML content
    body += """
        </table>
    </body>
    </html>
    """
    
    # Send the email
    send_email(subject, body)

if __name__ == '__main__':
    main()
