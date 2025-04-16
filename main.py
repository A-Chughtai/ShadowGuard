import numpy as np
import multiprocessing as mp
import pandas as pd
import csv
from collections import defaultdict, deque
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import random

class Node:
    def __init__(self, **kwargs):
        self.device_ip = kwargs["device_ip"]
        self.port_number = kwargs["port_number"]
        self.protocol_used = kwargs["protocol_used"]
        self.packets_received = int(kwargs["packets_received"]) if kwargs["packets_received"].isdigit() else 0
        self.packets_forwarded = int(kwargs["packets_forwarded"]) if kwargs["packets_forwarded"].isdigit() else 0
        self.unusual_login_attempts = int(kwargs["unusual_login_attempts"]) if kwargs["unusual_login_attempts"].isdigit() else 0
        self.resources_shared = kwargs["resources_shared"]
        self.device_type = kwargs["device_type"]
        self.cpu_usage = kwargs["cpu_usage"]
        self.memory_usage = kwargs["memory_usage"]
        self.storage_usage = kwargs["storage_usage"]
        self.device_uptime = kwargs["device_uptime"]
        self.os_version = kwargs["os_version"]
        self.connection_status = kwargs["connection_status"]
        self.latency = kwargs["latency"]
        self.error_rate = kwargs["error_rate"]
        self.bandwidth_usage = kwargs["bandwidth_usage"]
        self.data_transferred = int(kwargs["data_transferred"]) if kwargs["data_transferred"].isdigit() else 0
        self.security_level = kwargs["security_level"]
        self.vpn_status = kwargs["vpn_status"]
        self.firmware_version = kwargs["firmware_version"]
        self.last_maintenance_date = kwargs["last_maintenance_date"]
        self.label = kwargs["label"]
        self.attack_type = kwargs["attack_type"]




def read_and_segment_csv_to_nodes(file_path):
    # Define a dictionary to hold the server network arrays
    server_networks = defaultdict(list)

    # List to hold all Node objects
    node_objects = []

    # Analyze the CSV and read the data
    with open(file_path, 'r') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    # Create Node objects and identify server IP prefixes
    servers = set()
    for row in rows:
        node = Node(
            device_ip=row["device_ip"],
            port_number=row["port_number"],
            protocol_used=row["protocol_used"],
            packets_received=row["packets_received"],
            packets_forwarded=row["packets_forwarded"],
            unusual_login_attempts=row["unusual_login_attempts"],
            resources_shared=row["resources_shared"],
            device_type=row["device_type"],
            cpu_usage=row["cpu_usage"],
            memory_usage=row["memory_usage"],
            storage_usage=row["storage_usage"],
            device_uptime=row["device_uptime"],
            os_version=row["os_version"],
            connection_status=row["connection_status"],
            latency=row["latency"],
            error_rate=row["error_rate"],
            bandwidth_usage=row["bandwidth_usage"],
            data_transferred=row["data_transferred"],
            security_level=row["security_level"],
            vpn_status=row["vpn_status"],
            firmware_version=row["firmware_version"],
            last_maintenance_date=row["last_maintenance_date"],
            label=row["label"],
            attack_type=row["attack_type"]
        )
        node_objects.append(node)

        # Identify server IPs
        ip_parts = row['device_ip'].split('.')
        if ip_parts[-1] == '0':  # Server IP ends with '.0'
            servers.add('.'.join(ip_parts[:3]))

    # Segment Node objects into corresponding server networks
    for node in node_objects:
        ip_prefix = '.'.join(node.device_ip.split('.')[:3])
        if ip_prefix in servers:
            server_networks[ip_prefix].append(node)

    return server_networks, node_objects



def create_graph(server_networks):
    graphs = {}
    for server, nodes in server_networks.items():
        graph = defaultdict(list)
        for node in nodes:
            graph[node.device_ip]  # Initialize the node
            # Example: Adding some relations, you might update based on actual logic
            for other in nodes:
                if node.device_ip != other.device_ip:
                    graph[node.device_ip].append(other.device_ip)
        graphs[server] = graph
    return graphs


def evaluate_attack_conditions(node):
    if node.packets_received > 6000 or node.packets_forwarded > 3000:
        
        return True
    elif node.unusual_login_attempts > 3:
        
        return True
    elif node.data_transferred > 5000:
        
        return True
    else:
        
        return False
    

def bfs_search(graph, server_nodes):
    suspected_queue = deque()
    seen = set()  # Set to track unique nodes in the queue

    for start_node in server_nodes:
        visited = set()
        queue = deque([start_node.device_ip])

        while queue:
            current_ip = queue.popleft()
            if current_ip in visited:
                continue
            visited.add(current_ip)

            # Fetch node from its IP
            current_node = next(node for node in server_nodes if node.device_ip == current_ip)

            # Evaluate attack conditions
            if evaluate_attack_conditions(current_node) and current_node.device_ip not in seen:
                suspected_queue.append(current_node)
                seen.add(current_node.device_ip)  # Mark as seen

            # Enqueue neighbors
            for neighbor in graph[current_ip]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return suspected_queue


# device_ip
# port_number
# protocol_used
# packets_received
# packets_forwarded
# unusual_login_attempts
# resources_shared
# device_type
# cpu_usage
# memory_usage
# storage_usage
# device_uptime
# os_version
# connection_status
# latency
# error_rate
# bandwidth_usage
# data_transferred
# security_level
# vpn_status
# firmware_version
# last_maintenance_date
# label
# attack_type



def predict(sus):
    
    model = load_model('trained_model.keras')
    malicious = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []
    x8 = []
    x9 = []
    x10 = []
    x11 = []

    for i in range (len(sus)):
        x1.append(sus[i].packets_received)
        x2.append(sus[i].packets_forwarded)
        x3.append(sus[i].unusual_login_attempts)
        x4.append(sus[i].resources_shared)
        x5.append(sus[i].cpu_usage)
        x6.append(sus[i].memory_usage)
        x7.append(sus[i].storage_usage)
        x8.append(sus[i].device_uptime)
        x9.append(sus[i].latency)
        x10.append(sus[i].bandwidth_usage)
        x11.append(sus[i].data_transferred)
    
    x1 = np.array(x1)
    x2 = np.array(x2)
    x3 = np.array(x3)
    x4 = np.array(x4)
    x5 = np.array(x5)
    x6 = np.array(x6)
    x7 = np.array(x7)
    x8 = np.array(x8)
    x9 = np.array(x9)
    x10 = np.array(x10)
    x11 = np.array(x11)


    x1 = x1.astype(float)
    x2 = x2.astype(float)
    x3 = x3.astype(float)
    x4 = x4.astype(float)
    x5 = x5.astype(float)
    x6 = x6.astype(float)
    x7 = x7.astype(float)
    x8 = x8.astype(float)
    x9 = x9.astype(float)
    x10 = x10.astype(float)
    x11 = x11.astype(float)

    

    x = np.array([x8, x9, x10, x1, x2, x11, x4, x5, x6, x7, x3])
    x = np.hstack([arr.reshape(-1, 1) for arr in x])

    scaler = MinMaxScaler()
    x = scaler.fit_transform(x)
    
    
    y = model.predict(x)

    for i in range (len(sus)):
        if(y[i] > 0.5):
            malicious.append(sus[i].device_ip)
    

    return malicious


def remove_malicious_nodes(graphs, sus_ips):
    for i in range(len(sus_ips)):
        ip = sus_ips[i]

        for subnet, graph in graphs.items():
            # If the node exists in the graph for this subnet, remove it
            if ip in graph:
                # Remove the node from its neighbors' adjacency lists
                for neighbor in graph[ip]:
                    if ip in graph.get(neighbor, []):
                        graph[neighbor].remove(ip)
                # Finally, remove the node itself from the graph
                del graph[ip]
                print(f"Node {ip} found in subnet {subnet} and removed!!!")


def main():
    file_path = "data3.csv"
    server_networks, all_nodes = read_and_segment_csv_to_nodes(file_path)

    graphs = create_graph(server_networks)

    suspected_attacks = deque()
    seen_nodes = set()
    
    for server, nodes in server_networks.items():
        unique_attacks = bfs_search(graphs[server], nodes)
        for attack in unique_attacks:
            if attack.device_ip not in seen_nodes:
                suspected_attacks.append(attack)
                seen_nodes.add(attack.device_ip)

    # Display suspected attacks
    # print("Suspected Attack Nodes:")
    # for node in suspected_attacks:
    #     print(f"Device IP: {node.device_ip}, Attack Type: {node.attack_type}")
    
    
    # bhai ye sus attacks par prediction wala model laga kar aik array return kar raha hoon
    # array mae jo bhi howa wo graph sae khatam kar doon ga
    malicious = predict(suspected_attacks)

    print (f'The number of malicious nodes idnetified: {len(malicious)}')

    
    remove_malicious_nodes(graphs, malicious)


if __name__ == "__main__":
    main()

