import os  
path = r'D:\New folder (2)\New folder\smart-tire-analyzer\ai_model\hybrid_torch\trainer.py'  
f = open(path, 'r', encoding='utf-8')  
s = f.read()  
f.close()  
s = s.replace('reduction=\" "\none\\)', 'reduction=\none\)')  
f = open(path, 'w', encoding='utf-8')  
f.write(s)  
f.close() 
