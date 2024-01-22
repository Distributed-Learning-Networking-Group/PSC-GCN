# import os
# import sys
import netifaces
"""
# orignal version

routingGateway = netifaces.gateways()['default'][netifaces.AF_INET][0]
routingNicName = netifaces.gateways()['default'][netifaces.AF_INET][1]

for interface in netifaces.interfaces():
    if interface == routingNicName:
        # print netifaces.ifaddresses(interface)
        routingNicMacAddr = netifaces.ifaddresses(interface)[netifaces.AF_LINK][0]['addr']
        try:
            routingIPAddr = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['addr']
            # TODO(Guodong Ding) Note: On Windows, netmask maybe give a wrong result in 'netifaces' module.
            routingIPNetmask = netifaces.ifaddresses(interface)[netifaces.AF_INET][0]['netmask']
        except KeyError:
            pass

display_format = '%-30s %-20s'
print(display_format % ("Routing Gateway:", routingGateway))
print(display_format % ("Routing NIC Name:", routingNicName))
print(display_format % ("Routing NIC MAC Address:", routingNicMacAddr))
print(display_format % ("Routing IP Address:", routingIPAddr))
print(display_format % ("Routing IP Netmask:", routingIPNetmask))
"""


def getNicName():
    return netifaces.gateways()['default'][netifaces.AF_INET][1]

print(getNicName())