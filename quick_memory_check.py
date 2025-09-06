import psutil

print("MEMORY CLEANUP CHECK")
print("=" * 30)

# System memory
memory = psutil.virtual_memory()
print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
print(f"Available: {memory.available / (1024**3):.1f} GB")
print(f"Used: {memory.used / (1024**3):.1f} GB")
print(f"Usage: {memory.percent:.1f}%")
print()

print("TOP MEMORY CONSUMERS (>100MB):")
print("-" * 40)

processes = []
for proc in psutil.process_iter(['pid', 'name', 'memory_info']):
    try:
        memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
        if memory_mb > 100:  # Only show processes using >100MB
            processes.append((proc.info['name'], proc.info['pid'], memory_mb))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        continue

# Sort by memory usage
processes.sort(key=lambda x: x[2], reverse=True)

for name, pid, memory_mb in processes[:15]:
    print(f"{name:<25} PID {pid:<8} {memory_mb:>8.1f} MB")

print()
print("POTENTIAL CLEANUP CANDIDATES:")
print("-" * 35)

# Common applications that can be safely closed
cleanup_candidates = ['chrome', 'firefox', 'edge', 'teams', 'slack', 'discord', 
                     'spotify', 'steam', 'notepad++', 'winrar', 'vlc']

found_candidates = []
for name, pid, memory_mb in processes:
    name_lower = name.lower()
    for candidate in cleanup_candidates:
        if candidate in name_lower and memory_mb > 50:
            found_candidates.append((name, pid, memory_mb))
            break

if found_candidates:
    print("Consider closing these applications:")
    for name, pid, memory_mb in found_candidates:
        print(f"  {name} (PID {pid}): {memory_mb:.1f} MB")
    
    total_freeable = sum(x[2] for x in found_candidates)
    print(f"\nPotential memory to free: {total_freeable/1024:.1f} GB")
else:
    print("No obvious cleanup candidates found")
