import debugpy

debugpy.listen(("0.0.0.0", 9000))
print("waiting ...")
debugpy.wait_for_client()
