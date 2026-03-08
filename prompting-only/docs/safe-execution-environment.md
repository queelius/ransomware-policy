# Safe Execution Environment for Ransomware Telemetry Generation

## Overview

This document provides detailed instructions for setting up a safe, isolated environment to generate realistic ransomware telemetry without risk to production systems.

## Safety Principles

1. **Complete Isolation**: No connection to production networks
2. **Snapshot/Restore**: Always work with restorable VMs
3. **Air Gap**: Physical or logical separation from sensitive data
4. **Monitoring**: Full telemetry capture without risk
5. **Controlled Execution**: Never run actual ransomware, only simulations

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Host Machine                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │            Isolated Virtual Network              │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │  │
│  │  │  Attack  │  │  Victim  │  │   Telemetry  │  │  │
│  │  │    VM    │──│   VMs    │──│   Collector  │  │  │
│  │  └──────────┘  └──────────┘  └──────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Option 1: Atomic Red Team Setup (Recommended)

### Requirements
- VMware Workstation/VirtualBox/Hyper-V
- Windows 10/11 VM (4GB RAM, 40GB disk)
- Ubuntu 22.04 VM (2GB RAM, 20GB disk)
- Isolated virtual network

### Setup Steps

#### 1. Create Isolated Network

```bash
# VirtualBox example
VBoxManage natnetwork add --netname ransomware-lab \
  --network "10.0.2.0/24" --enable --dhcp off

# VMware example
# Use Virtual Network Editor to create custom network
```

#### 2. Configure Victim VM (Windows)

```powershell
# Install Sysmon for telemetry
Invoke-WebRequest -Uri https://github.com/SwiftOnSecurity/sysmon-config/raw/master/sysmonconfig-export.xml -OutFile sysmon.xml
sysmon64 -accepteula -i sysmon.xml

# Install Atomic Red Team
IEX (IWR 'https://raw.githubusercontent.com/redcanaryco/invoke-atomicredteam/master/install-atomicredteam.ps1' -UseBasicParsing);
Install-AtomicRedTeam -getAtomics

# Configure Windows logging
wevtutil set-log Microsoft-Windows-Sysmon/Operational /enabled:true /retention:false /maxsize:1073741824
auditpol /set /category:"Detailed Tracking" /success:enable /failure:enable
```

#### 3. Configure Victim VM (Linux)

```bash
# Install auditd
sudo apt-get update
sudo apt-get install -y auditd audispd-plugins

# Configure audit rules for ransomware detection
cat << EOF | sudo tee /etc/audit/rules.d/ransomware.rules
# File access monitoring
-w /home/ -p wa -k file_changes
-w /etc/ -p wa -k system_changes

# Process execution
-a exit,always -F arch=b64 -S execve -k process_execution
-a exit,always -F arch=b32 -S execve -k process_execution

# Network connections
-a exit,always -F arch=b64 -S connect -k network_connections
EOF

sudo service auditd restart
```

#### 4. Setup Telemetry Collector

```bash
# Create collector VM or use host
mkdir -p /opt/telemetry-collector
cd /opt/telemetry-collector

# Install collection tools
pip install elasticsearch logstash

# Configure Elastic Stack for telemetry ingestion
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  docker.elastic.co/elasticsearch/elasticsearch:8.11.0
```

#### 5. Run Safe Simulations

```powershell
# On Windows victim VM
# Run specific ransomware technique simulations
Invoke-AtomicTest T1486 -TestNumbers 1  # File encryption simulation
Invoke-AtomicTest T1490 -TestNumbers 1  # Shadow copy deletion
Invoke-AtomicTest T1059.001 -TestNumbers 3  # PowerShell execution

# These create telemetry without actual harm
```

## Option 2: Infection Monkey Setup

### Installation

```bash
# On collector VM
wget https://github.com/guardicore/monkey/releases/latest/download/infection_monkey_docker.tar.gz
docker load -i infection_monkey_docker.tar.gz

docker run -d \
  --name monkey-island \
  --network=ransomware-lab \
  -p 5000:5000 \
  guardicore/monkey-island:latest
```

### Configuration

```json
{
  "ransomware": {
    "enabled": true,
    "encryption_path": "/tmp/test_files",
    "file_extension": ".m0nk3y",
    "leave_readme": true
  },
  "telemetry": {
    "export_to": "elasticsearch",
    "endpoint": "http://10.0.2.100:9200"
  }
}
```

## Option 3: Custom Sandbox with Cuckoo

### Setup Cuckoo Sandbox

```bash
# Install Cuckoo
pip3 install cuckoo

# Initialize
cuckoo init
cuckoo community

# Configure for ransomware analysis
cat << EOF > ~/.cuckoo/conf/cuckoo.conf
[cuckoo]
machinery = virtualbox
memory_dump = yes
terminate_processes = yes

[timeouts]
default = 300
critical = 600
vm_state = 60
EOF
```

### Safe Sample Execution

```python
# submit_safe_sample.py
from cuckoo.core.database import Database
from cuckoo.core.submit import Submit

# Only submit known safe simulators
SAFE_SAMPLES = [
    "/samples/ransomware_simulator.exe",  # Your simulator
    "/samples/atomic_test.ps1"            # Atomic test
]

db = Database()
submit = Submit()

for sample in SAFE_SAMPLES:
    task_id = submit.add_path(sample, options={
        "timeout": 300,
        "enforce_timeout": True,
        "memory": True,
        "procmemdump": True
    })
    print(f"Submitted {sample} as task {task_id}")
```

## Telemetry Collection Pipeline

### Windows Event Collection

```xml
<!-- Sysmon config for ransomware detection -->
<Sysmon schemaversion="4.90">
  <EventFiltering>
    <ProcessCreate onmatch="include">
      <CommandLine condition="contains">vssadmin</CommandLine>
      <CommandLine condition="contains">bcdedit</CommandLine>
      <CommandLine condition="contains">wbadmin</CommandLine>
      <Image condition="contains">powershell.exe</Image>
    </ProcessCreate>

    <FileCreate onmatch="include">
      <TargetFilename condition="end with">.encrypted</TargetFilename>
      <TargetFilename condition="end with">.locked</TargetFilename>
      <TargetFilename condition="contains">README</TargetFilename>
    </FileCreate>
  </EventFiltering>
</Sysmon>
```

### Linux Audit Collection

```bash
# Parse audit logs for ransomware patterns
ausearch -k file_changes --format json | \
  jq 'select(.exe | contains(["gpg", "openssl", "7z"]))'

# Monitor file entropy changes
inotifywait -m -r /home --format '%w%f %e' | while read file event; do
  if [[ "$event" == *"MODIFY"* ]]; then
    entropy=$(ent "$file" | grep "Entropy" | awk '{print $3}')
    echo "$(date): $file entropy: $entropy"
  fi
done
```

## Safety Checklist

Before running any simulation:

- [ ] VMs are on isolated network with no external access
- [ ] Production data is not accessible from test environment
- [ ] VM snapshots taken before each test
- [ ] Telemetry collection is functioning
- [ ] Kill switch is ready (ability to instantly stop/revert VMs)
- [ ] No actual ransomware binaries are used
- [ ] Team is notified of testing window

## Data Export

After simulation:

```python
# export_telemetry.py
import json
from elasticsearch import Elasticsearch

es = Elasticsearch(['localhost:9200'])

# Query telemetry data
response = es.search(
    index="sysmon-*",
    body={
        "query": {"match_all": {}},
        "size": 10000
    }
)

# Export to JSONL for model training
with open('telemetry_export.jsonl', 'w') as f:
    for hit in response['hits']['hits']:
        f.write(json.dumps(hit['_source']) + '\n')
```

## Validation

Verify telemetry quality:

```python
# validate_telemetry.py
import json

required_event_types = [
    'process_create',
    'file_write',
    'network_connection',
    'registry_modify'
]

with open('telemetry_export.jsonl') as f:
    events = [json.loads(line) for line in f]

    event_types = set(e.get('event_type') for e in events)

    print(f"Collected event types: {event_types}")
    print(f"Missing required types: {set(required_event_types) - event_types}")

    # Check for ransomware indicators
    indicators = [
        e for e in events
        if 'vssadmin' in str(e.get('command_line', ''))
        or e.get('entropy_after', 0) > 7.5
    ]

    print(f"Found {len(indicators)} ransomware indicators")
```

## Cleanup

After testing:

```bash
# Revert VMs to clean snapshots
VBoxManage snapshot "Victim-Win10" restore "Clean"
VBoxManage snapshot "Victim-Ubuntu" restore "Clean"

# Clear telemetry data
curl -X DELETE "localhost:9200/sysmon-*"

# Archive collected data
tar -czf telemetry_$(date +%Y%m%d).tar.gz telemetry_export.jsonl
```

## Additional Resources

- [Atomic Red Team](https://github.com/redcanaryco/atomic-red-team)
- [Infection Monkey Docs](https://www.akamai.com/infectionmonkey)
- [Cuckoo Sandbox](https://cuckoosandbox.org/)
- [Sysmon Configuration](https://github.com/SwiftOnSecurity/sysmon-config)
- [MITRE ATT&CK](https://attack.mitre.org/)

## Important Warnings

⚠️ **NEVER**:
- Run actual ransomware samples outside of proper malware analysis labs
- Connect test environment to production networks
- Store sensitive data in test VMs
- Skip the snapshot/restore process
- Disable safety controls to "see what happens"

✅ **ALWAYS**:
- Use simulation tools rather than real malware
- Maintain complete isolation
- Document what you're testing
- Have a recovery plan
- Follow your organization's security policies