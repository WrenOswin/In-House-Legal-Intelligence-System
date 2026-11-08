PY="/c/Users/wreno/OneDrive/Desktop/LLMs/Assignments/.venv/Scripts/python.exe"
for spec in 'D_d560 lora 3e-4' 'D_d560 lora 1e-3' 'C_d560 houlsby 3e-4' 'C_d560 houlsby 1e-3' 'B_d512 pfeiffer 3e-4' 'B_d512 pfeiffer 1e-3' 'A_d560 full 5e-5' 'A_d560 full 2e-4'; do
  set -- $spec
  echo "### $spec"
  $PY run.py "{\"exp\":\"pilot_lr\",\"backbone\":\"$1\",\"method\":\"$2\",\"lr\":$3,\"max_train_docs\":800,\"epochs\":1,\"duty\":1.0,\"make_submission\":false,\"val_gen_docs\":32,\"max_new\":256}" 2>&1 | grep -E "params:|val loss|val generation|Error|error" | tail -5
done
