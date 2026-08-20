# Blended trigger image

把 Blended 触发器用的图案（如 hello-kitty）放在这里，默认路径：

    attack/triggers/hello_kitty.png

或在 config 里设置 `backdoor.blended_image: <你的路径>`。

图片会被自动 resize 到 `data.img_size`（CIFAR-10 为 32×32）并按 CIFAR-10
逐通道标准化，再以 `backdoor.blended_alpha`（默认 0.2）与干净图混合。

BadNet 触发器（默认 `backdoor.trigger: badnet`）不需要任何图片文件。
