# Adivyanta 360M roast adapter

This is a 4,014,080-trainable-parameter LoRA adapter for [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), revision `a10cc1512eabd3dde888204e902eca88bddb4951`. It is **not** a standalone 360M weight file. The base model is Apache-2.0 licensed; this adapter was trained only on original examples written for Adivyanta. The separate scratch-trained 10M weights in this repository derive from CC BY-NC 4.0 EmpatheticDialogues and have noncommercial restrictions.

Training used 292 roast and control examples, with 36 held-out target replies. The base model's response-token validation loss was 4.842; the best adapter reached 3.742 at epoch 3. Epoch 4 worsened to 3.872, so epoch 3 is saved. The validation set shares topics with training, so these figures measure style adaptation more than broad generalization.

The adapter's raw output can still be repetitive, unfunny, or inappropriate for serious messages. Use the repository's `chat.py` for the routed chat experience; `chat_adivyanta.py --model-only` shows the raw adapter. The 40-prompt comparison and all raw outputs are in `benchmarks/`.
