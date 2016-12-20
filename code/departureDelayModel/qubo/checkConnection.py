#!/usr/bin/env python
import dwave_sapi2.remote as remote
import myToken

conn = remote.RemoteConnection(myToken.myUrl, myToken.myToken)
#conn = remote.RemoteConnection('socks5://127.0.0.1:12345', myToken.myToken)
print conn.solver_names()
