# Run this script with sudo privileges to register the cron job and start the server

service=billard-beamer-server

# relative to this file:
#templatefolder=../../templates/

echo "Installing ${service}..."


# create and activate a python virtual environment and install all required modules
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt


# create and install the systemd service
cp ./${service}.service /etc/systemd/system/${service}.service && echo "File succesfully moved"

systemctl daemon reload && echo "Systemctl daemon reloaded"
systemctl enable ${service}.service && echo "Service succesfully registered for startup"
systemctl start ${service}.service && echo "Service succesfully started"

systemctl status ${service}.service


# Output the current IP address:
echo "IP addresses:"
ip a | grep inet
echo "Note this IP address for use in the configuration of other modules! In most cases, it is beneficial to have it set as a static IP."