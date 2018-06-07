# data-science

# set up Azure VM:
To set up a VM for machine learning follow the  below steps: 
1)  Navigate to your Azure account (https://portal.azure.com/) 
2) Once logged in, select create a new resource and type: Data Science Virtual Machine for Linux Ubuntu CSP 
3) create username and password, you need them in next steps 
4) create group
5) select a machine with this (Standard D412 v2 ). 
 Note: Machine should have Jupiter and Python 3(or above) installed
6) Once the virtual machine generated, select  the overview tab and copy the IP address 
7) use Putty[1] to connect to the VM (using user/pass provided above)  
NOTE: 
Make sure the Azure SDK is installed on your created VM: 
#git clone https://github.com/Azure-Samples/cognitive-services-python-sdk-samples.git 
 
 You may run your source code on the Installed Jupyter in the instaled VM (https:{your-vm-ip-address:8000/})

