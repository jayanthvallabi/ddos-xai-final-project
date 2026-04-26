from scapy.all import sniff
import requests
import time

API_URL = "http://127.0.0.1:8000/predict"

packet_count = 0
src_bytes = 0
dst_bytes = 0
start_time = time.time()

def process_packet(packet):
    global packet_count, src_bytes, dst_bytes, start_time

    packet_count += 1
    pkt_len = len(packet)
    src_bytes += pkt_len

    duration = time.time() - start_time

    if packet_count % 10 == 0:
        sample = {
            "duration": duration,
            "src_bytes": src_bytes,
            "dst_bytes": dst_bytes,
            "count": packet_count,
            "srv_count": packet_count,
            "dst_host_count": packet_count,
            "dst_host_srv_count": packet_count,
            "wrong_fragment": 0,
            "urgent": 0,
            "hot": 0,
            "num_failed_logins": 0,
            "logged_in": 0,
            "num_compromised": 0,
            "root_shell": 0,
            "su_attempted": 0,
            "num_root": 0,
            "num_file_creations": 0,
            "num_access_files": 0,
            "num_outbound_cmds": 0,
            "is_host_login": 0,
            "is_guest_login": 0
        }

        try:
            r = requests.post(API_URL, json=sample)
            print(r.json())
        except Exception as e:
            print("API error:", e)

print("Capturing live packets...")
sniff(prn=process_packet, store=False)