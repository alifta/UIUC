# BFS
def simple_bfs(graph,source):
    """A fast BFS node generator"""
    adj = graph.adj
    seen = set()
    nextlevel = {source}
    while nextlevel:
        thislevel = nextlevel
        nextlevel = set()
        for v in thislevel:
            if v not in seen:
                seen.add(v)
                nextlevel.update(adj[v])
    return seen