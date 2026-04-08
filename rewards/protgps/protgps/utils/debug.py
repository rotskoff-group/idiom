def debug_vscode():
    """
    Since this requires listening on a local port, this will only work when VSCode is connected to the same machine.
    Before running, add the following to your launch.json:
        {"version":"0.2.0","configurations":[{"name":"Python: Remote Attach","type":"python","request":"attach","connect":{"host":"localhost","port":5678},"pathMappings":[{"localRoot":"${workspaceFolder}",
        "remoteRoot":"."}],"justMyCode":true}]}
    """
    import debugpy

    print("Waiting for VSCode debugger to attach...")
    debugpy.listen(5678)
    debugpy.wait_for_client()
    print("VSCode debugger attached!")
