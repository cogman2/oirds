from imposm.parser import OSMParser

class HighwayCounter(object):
   highways = dict()
   points  = dict()
   def ways(self, ways):
     for osmid, tags, refs in ways:
       isWater = ('water' in tags) or ('waterway' in tags) or ('coastline' in tags)
       if isWater:
          for x in refs:
            if not self.highways.has_key(x):
              self.highways[x] = list()
            self.highways[x].append(osmid)
   def nodes(self, nodes):
     for osmid, tags, ll in nodes:
       if
 self.highways.has_key(osmid):
          for k in self.highways[osmid]:
            if not self.points.has_key(k):
               self.points[k] = list()
            self.points[k].append(ll)

counter = HighwayCounter()

p = OSMParser(concurrency=4, ways_callback=counter.ways)
p.parse('sa.osm.pbf')

p = OSMParser(concurrency=4, nodes_callback=counter.nodes)
p.parse('sa.osm.pbf')
