"""
手动保存已训练的BC模型
在Python交互环境中运行
"""
import torch

# 如果你的bc_trainer对象还在内存中，运行：
# import torch
# torch.save(bc_trainer.policy.state_dict(), "./bc_models_imitation/bc_policy.pth")
# torch.save(bc_trainer.policy, "./bc_models_imitation/bc_policy.pt")

print("如果bc_trainer还在内存中，请在Python中运行:")
print(">>> import torch")
print(">>> torch.save(bc_trainer.policy.state_dict(), './bc_models_imitation/bc_policy.pth')")
print(">>> torch.save(bc_trainer.policy, './bc_models_imitation/bc_policy.pt')")
print("\n模型就会保存成功！")

