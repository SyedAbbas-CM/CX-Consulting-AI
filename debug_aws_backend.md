# AWS Backend Debug Commands

## 1. Check if the application is running
```bash
ps aux | grep python
ps aux | grep start.py
```

## 2. Check if port 8000 is listening
```bash
sudo netstat -tlnp | grep 8000
# or
sudo ss -tlnp | grep 8000
```

## 3. Check application logs
```bash
tail -f app.log
# or if no app.log
journalctl -u your-service-name -f
```

## 4. Check system resources
```bash
free -h          # RAM usage
df -h            # Disk space
top              # CPU usage
```

## 5. Try starting the application manually
```bash
cd /path/to/your/app
source venv/bin/activate
python start.py
```

## 6. Check if firewall is blocking
```bash
sudo ufw status
# Check security group in AWS console
```

## 7. Test local connection on the server
```bash
curl localhost:8000/api/health
```

## 8. Check nginx status (if using reverse proxy)
```bash
sudo systemctl status nginx
sudo nginx -t
```

## Common Issues:
- Application crashed due to model loading issues
- Out of memory (check with `free -h`)
- Disk full (check with `df -h`)
- Port already in use
- Firewall blocking external connections
- Security group not allowing port 8000
