#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spring 2017
@author: YuJungWang
"""

import bpy
import sys
import bmesh
import os
import datetime
import csv
import random
#import numpy
from mathutils import Vector



##--------------------Finding all objects in camera view--------------------##  
  
class ObjectsInCameraView:
	"""
	Return all objects in camera view border
	"""
	def __init__(self):
		from bpy import context
		scene = context.scene
		origin = scene.camera.matrix_world.to_translation()
		planes = self.camera_as_planes(scene, scene.camera)
		self.objects_in_view = self.objects_in_planes(scene.objects, planes, origin)
		
		
	def camera_as_planes(self, scene, obj):
		"""
		Return planes in world-space which represent the camera view bounds.
		"""
		from mathutils.geometry import normal

		camera = obj.data
		# normalize to ignore camera scale
		matrix = obj.matrix_world.normalized()
		frame = [matrix * v for v in camera.view_frame(scene)]
		origin = matrix.to_translation()

		planes = []
		from mathutils import Vector
		is_persp = (camera.type != 'ORTHO')
		for i in range(4):
			# find the 3rd point to define the planes direction
			if is_persp:
				frame_other = origin
			else:
				frame_other = frame[i] + matrix.col[2].xyz

			n = normal(frame_other, frame[i - 1], frame[i])
			d = -n.dot(frame_other)
			planes.append((n, d))

		if not is_persp:
			# add a 5th plane to ignore objects behind the view
			n = normal(frame[0], frame[1], frame[2])
			d = -n.dot(origin)
			planes.append((n, d))

		return planes


	def side_of_plane(self, p, v):
		return p[0].dot(v) + p[1]


	def is_segment_in_planes(self, p1, p2, planes):
		dp = p2 - p1

		p1_fac = 0.0
		p2_fac = 1.0

		for p in planes:
			div = dp.dot(p[0])
			if div != 0.0:
				t = -self.side_of_plane(p, p1)
				if div > 0.0:
					# clip p1 lower bounds
					if t >= div:
						return False
					if t > 0.0:
						fac = (t / div)
						p1_fac = max(fac, p1_fac)
						if p1_fac > p2_fac:
							return False
				elif div < 0.0:
					# clip p2 upper bounds
					if t > 0.0:
						return False
					if t > div:
						fac = (t / div)
						p2_fac = min(fac, p2_fac)
						if p1_fac > p2_fac:
							return False

		## If we want the points
		# p1_clip = p1.lerp(p2, p1_fac)
		# p2_clip = p1.lerp(p2, p2_fac)        
		return True


	def point_in_object(self, obj, pt):
		xs = [v[0] for v in obj.bound_box]
		ys = [v[1] for v in obj.bound_box]
		zs = [v[2] for v in obj.bound_box]
		pt = obj.matrix_world.inverted() * pt
		return (min(xs) <= pt.x <= max(xs) and
				min(ys) <= pt.y <= max(ys) and
				min(zs) <= pt.z <= max(zs))


	def object_in_planes(self, obj, planes):
		from mathutils import Vector

		matrix = obj.matrix_world
		box = [matrix * Vector(v) for v in obj.bound_box]
		for v in box:
			if all(self.side_of_plane(p, v) < 0.0 for p in planes):
				# one point was in all planes
				return True

		# possible one of our edges intersects
		edges = ((0, 1), (0, 3), (0, 4), (1, 2),
				 (1, 5), (2, 3), (2, 6), (3, 7),
				 (4, 5), (4, 7), (5, 6), (6, 7))
		if any(self.is_segment_in_planes(box[e[0]], box[e[1]], planes)
			   for e in edges):
			return True


		return False


	def objects_in_planes(self, objects, planes, origin):
		"""
		Return all objects which are inside (even partially) all planes.
		"""
		return [obj for obj in objects
				if self.point_in_object(obj, origin) or
				   self.object_in_planes(obj, planes)]

				   
				
##--------------------Counting the vertices, edges, faces for a given object--------------------##

# Used to store information related to a camera setting for a given object
# Create the new coordinate such that let the coordinate of camera be (0, 0, 0), 
# And the normal of camera face is z-axis, behind the camera is negative
class CameraSetting:
	def __init__( self, scene, camera, obj ):
		self.camera = camera
		#Z axis of the camera translate in world (Z axis is opposite to view)
		self.zInWorld = Vector( (0, 0, 1) )
		self.zInWorld.rotate( self.camera.matrix_world.to_euler() )
		#Camera parameters
		#Matrix to convert from object coordinate to camera coordinates
		self.toCameraMatrix = camera.matrix_world.inverted() * obj.matrix_world
		#Matrix to convert from camera coordinate to object coordinates
		self.toObjectMatrix = obj.matrix_world.inverted() * camera.matrix_world
		#The frame is composed of the coordinates in the camera view
		self.frame = [v / v.z for v in camera.data.view_frame(scene=scene)]
		#Get the X, Y corners
		self.minX = min( v.x for v in self.frame )
		self.maxX = max( v.x for v in self.frame )
		self.minY = min( v.y for v in self.frame )
		self.maxY = max( v.y for v in self.frame )
		#Precalculations to avoid to repeat them when applied to the model
		self.deltaX = self.maxX - self.minX
		self.deltaY = self.maxY - self.minY
		self.offsetX = self.minX / self.deltaX
		self.offsetY = self.minY / self.deltaY
		
		
	#Calculate projected coordinateds from the object coordinates
	def CalcProjected( self, objCo ):
		#Object coordinate in camera view
		camCo = self.toCameraMatrix * objCo
		#Z is "inverted" as camera view is pointing to -Z of the camera
		z = -camCo.z
		try:
			#Translates x and y to projected coordinates
			x = (camCo.x / (self.deltaX * z)) - self.offsetX        
			y = (camCo.y / (self.deltaY * z)) - self.offsetY        
			return x, y, z
		except:
			#In case Z is zero
			return 0.5, 0.5, 0
		
		
	# The EDGE will across the camera view border by: 
	# 1. If exist tX in [0,1] such that x1+tX(x2-x1)=minX(z1+t(z2-z1)) or x1+tX(x2-x1)=maxX(z1+t(z2-z1))
	# 2. Or if exist t2 in [0,1] such that y1+tY(y2-y1)=minY(z1+t(z2-z1)) or y1+tY(y2-y1)=maxY(z1+t(z2-z1))
	def edgeCheckAcross( self, cam, objCo1, objCo2):
		#Object coordinate in camera view
		camCo1 = self.toCameraMatrix * objCo1
		camCo2 = self.toCameraMatrix * objCo2
		#Z is "inverted" as camera view is pointing to -Z of the camera
		z1 = -camCo1.z
		z2 = -camCo2.z
		# deltaX1toX2 = (x2-x1)
		deltaX1toX2 = camCo2.x - camCo1.x
		# deltaY1toY2 = (y2-y1)
		deltaY1toY2 = camCo2.y - camCo1.y
		#print("x, y, z", camCo.x, camCo.y, camCo.z)	
		
		try:	
			# Solve tminX, tmaxX, tminY, tmaxY
			tminX = (camCo1.x - self.minX * z1) / (self.minX * (z2 - z1) - camCo2.x + camCo1.x)
			tmaxX = (camCo1.x - self.maxX * z1) / (self.maxX * (z2 - z1) - camCo2.x + camCo1.x)
			tminY = (camCo1.y - self.minY * z1) / (self.minY * (z2 - z1) - camCo2.y + camCo1.y)
			tmaxY = (camCo1.y - self.maxY * z1) / (self.maxY * (z2 - z1) - camCo2.y + camCo1.y)
			
			# If the vector from v1 touch the border of camera view at p(in X or Y ) and p is in the view distance
			if( (0 <= tminX <= 1 and cam.data.clip_start <= (z1 + tminX*(z2-z1)) <= cam.data.clip_end) or
				(0 <= tmaxX <= 1 and cam.data.clip_start <= (z1 + tmaxX*(z2-z1)) <= cam.data.clip_end) or 
				(0 <= tminY <= 1 and cam.data.clip_start <= (z1 + tminY*(z2-z1)) <= cam.data.clip_end) or 
				(0 <= tmaxY <= 1 and cam.data.clip_start <= (z1 + tmaxY*(z2-z1)) <= cam.data.clip_end) ):
		
				return True
			return False
		except:
			#In case devider is zero
			return True

	# Project the face to the end plane of the camera.
	def ProjVertToEndPlane( self, cam, vList):
		# the minimum and maximum of x and y in camera coordinate
		vInEndPlaneList = []
		vi = 0
		# Transform all vertices of the face to projected vertices and save them in vList
		for v in vList:
			#print("original v: ", v)
			# Object coordinate in camera view
			camCo = self.toCameraMatrix * Vector(v)
			# Z is "inverted" as camera view is pointing to -Z of the camera
			oldDistance = abs(camCo.z)
			# the coordinate of the projected vertice
			x = camCo.x * cam.data.clip_end / oldDistance
			y = camCo.y * cam.data.clip_end / oldDistance
			
			if ( x < self.minX * cam.data.clip_end ):
				x = self.minX * cam.data.clip_end
			#else:
			#	vInEndPlaneList.append( ( x, y ) )
			if ( x > self.maxX * cam.data.clip_end ):
				x = self.maxX * cam.data.clip_end
			#else:
			#	vInEndPlaneList.append( ( x, y ) )
			if ( y < self.minY * cam.data.clip_end ):
				y = self.minY * cam.data.clip_end
			#else:
			#	vInEndPlaneList.append( ( x, y ) )
			if ( y > self.maxY * cam.data.clip_end ): 
				y = self.maxY * cam.data.clip_end
			#else:
			#	vInEndPlaneList.append( ( x, y ) )
			
			vInEndPlaneList.append( ( x, y ) )
			#print("v after projecting: ", ( x, y ))
		
		return vInEndPlaneList
		
	
	# Project the face to the end plane of the camera, 
	# and return the list of the minimum and maximum of x and y of projected face.
	def ProjVertToEndPlaneFindRec( self, cam, vList):
		# the minimum and maximum of x and y in camera coordinate
		min_max_Of_xy = []
		vi = 0
		# Transform all vertices of the face to projected vertices and save them in vList
		for v in vList:
			# Object coordinate in camera view
			camCo = self.toCameraMatrix * Vector(v)
			# Z is "inverted" as camera view is pointing to -Z of the camera
			oldDistance = -camCo.z
			# the coordinate of the projected vertice
			x = camCo.x * cam.data.clip_end / oldDistance
			y = camCo.y * cam.data.clip_end / oldDistance
			newCamCo = Vector((x, y, -cam.data.clip_end))
			# Initiate the values of min_max_Of_xy and the vertices of vListInCam
			if min_max_Of_xy == []:
				min_max_Of_xy.extend([newCamCo[0] for i in range(2)])
				min_max_Of_xy.extend([newCamCo[1] for i in range(2)])
				if ( min_max_Of_xy[0] < self.minX * cam.data.clip_end ):
					min_max_Of_xy[0] = self.minX * cam.data.clip_end
				if ( min_max_Of_xy[1] > self.maxX * cam.data.clip_end ):
					min_max_Of_xy[1] = self.maxX * cam.data.clip_end
				if ( min_max_Of_xy[2] < self.minY * cam.data.clip_end ):
					min_max_Of_xy[2] = self.minY * cam.data.clip_end
				if ( min_max_Of_xy[3] > self.maxY * cam.data.clip_end ): 
					min_max_Of_xy[3] = self.maxY * cam.data.clip_end
			# go through all vertices in this face
			else:
				if (newCamCo[0] < min_max_Of_xy[0] 
					and newCamCo[0] < self.minX * cam.data.clip_end):
					min_max_Of_xy[0] = self.minX * cam.data.clip_end
				elif newCamCo[0] < min_max_Of_xy[0]:
					min_max_Of_xy[0] = newCamCo[0]
				elif (newCamCo[0] > min_max_Of_xy[1] 
					and newCamCo[0] > self.maxX * cam.data.clip_end):
					min_max_Of_xy[1] = self.maxX * cam.data.clip_end
				elif newCamCo[0] > min_max_Of_xy[1]:
					min_max_Of_xy[1] = newCamCo[0]
				if (newCamCo[1] < min_max_Of_xy[2] 
					and newCamCo[1] < self.minY * cam.data.clip_end):
					min_max_Of_xy[2] = self.minY * cam.data.clip_end
				elif newCamCo[1] < min_max_Of_xy[2]:
					min_max_Of_xy[2] = newCamCo[1]
				elif (newCamCo[1] > min_max_Of_xy[3] 
					and newCamCo[1] > self.maxY * cam.data.clip_end):
					min_max_Of_xy[3] = self.maxY * cam.data.clip_end
				elif newCamCo[1] > min_max_Of_xy[3]:
					min_max_Of_xy[3] = newCamCo[1]
					
		# Return the minimum and maximum of x and y in camera coordinate
		return min_max_Of_xy
	
	
##--------------------Calculating Coverage--------------------##  
  
class CalcCoverage:			
	'''
		When called returns a list of points that forms a convex hull around 

		the vList Given
	'''
	
	def get_hull_points(self, vList):

		# get the min, and max from the list of verts
		min, max = self.get_min_max_x(vList)

		hullvs = self.quickhull(vList, min, max)

		hullvs = hullvs + self.quickhull(vList, max, min)

		return hullvs 

	'''
		Does the sorting for the quick hull sorting algorithm
	'''
	def quickhull(self, vList, min, max):
		left_of_line_vs = self.get_points_left_of_line(min, max, vList)

		ptC = self.point_max_from_line(min, max, left_of_line_vs)

		if len(ptC) < 1:
			return [max]

		hullvs = self.quickhull(left_of_line_vs, min, ptC)

		hullvs = hullvs + self.quickhull(left_of_line_vs, ptC, max)

		return hullvs

	'''
		Reterns all points that a LEFT of a line start->end
	'''
	def get_points_left_of_line(self, start, end, vList):
		pts = []

		for pt in vList:
			if self.isCCW(start, end, pt):
				pts.append(pt)

		return pts

	'''
		Returns the maximum point from a line start->end
	'''
	def point_max_from_line(self, start, end, points):
		max_dist = 0

		max_point = []

		for point in points:
			if point != start and point != end:
				dist = self.distance(start, end, point)
				if dist > max_dist:
					max_dist = dist
					max_point = point

		return max_point

	def get_min_max_x(self, list_pts):
		min_x = float('inf')
		max_x = 0
		min_y = 0
		max_y = 0

		for x,y in list_pts:
			if x < min_x:
				min_x = x
				min_y = y
			if x > max_x:
				max_x = x
				max_y = y

		return [min_x,min_y], [max_x,max_y]

	'''
		Given a line of start->end, will return the distance that
		point, pt, is from the line.
	'''
	def distance(self, start, end, pt): # pt is the point
		x1, y1 = start
		x2, y2 = end
		x0, y0 = pt
		nom = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
		denom = ((y2 - y1)**2 + (x2 - x1) ** 2) ** 0.5
		result = nom / denom
		return result
		
		
	def isCCW(self, start, end, pt):
		a1 = end[0] - start[0]
		a2 = end[1] - start[1]
		b1 = pt[0] - start[0]
		b2 = pt[1] - start[1]
		
		outer = a1*b2 - a2*b1
		if ( outer < 0 ):
			return True
		else:
			return False
			
			
			
# Counting the vertices, edges for a given object in camera view
def CountVerticeForObject( scene, obj, cam ):
	#matrix_world = obj.matrix_world
	#rotation_world = matrix_world.to_euler()

	#vertices = obj.data.vertices
	camSetting = CameraSetting( scene, cam, obj )

	vertexCount = 0
	vlist = []
	#Go through all polygons
	for v in obj.data.vertices:
		x, y, z = camSetting.CalcProjected( v.co )
		#print("z = ", z, "start : ",cam.data.clip_start, "end : ", cam.data.clip_end)
		if(0 <= x <= 1 and 0 <= y <= 1 and cam.data.clip_start <= z <= cam.data.clip_end):
			vertexCount += 1
			vlist.append( ( obj, (v.co.x, v.co.y, v.co.z) ) )
	
	return vertexCount, vlist
	
	
	
# Counting the edges for a given object in camera view
def CountEdgeForObject( scene, obj, cam ):
	camSetting = CameraSetting( scene, cam, obj )
	# Enter EDIT mode of this object since bmesh only can be acess in EDIT mode
	if obj.type == 'MESH':
		scene.objects.active = obj
		bpy.ops.object.mode_set(mode = 'EDIT')
	
	me = obj.data

	# Acess bmesh of this mesh
	if me.is_editmode:
		bm = bmesh.from_edit_mesh(me)
	else:
		bm = bmesh.new()
		bm.from_mesh(me)
	'''
	if obj.type == 'MESH':
		scene.objects.active = obj
		bpy.ops.object.mode_set(mode = 'EDIT')
		#bpy.ops.mesh.remove_doubles(threshold=0.0001)
	
	me = obj.data
	bm = bmesh.from_edit_mesh(me)
	'''
	edgeCount = 0
	# Go through all polygons
	for e in bm.edges:
		addEdge = 0
		# If one of the collected vertices on the selected edge is in camera view border, 
		# then count the edge. Otherwise, don't count it.
		for v in e.verts :
			x, y, z = camSetting.CalcProjected( v.co )
			if (0 <= x <= 1 and 0 <= y <= 1 and cam.data.clip_start <= z <= cam.data.clip_end) :
				addEdge = 1
				break
		# Check whether the edge will across the camera view border, 
		# if yes, count the edge. Otherwise, don't count it.
		if addEdge == 0:
			if (camSetting.edgeCheckAcross( cam, e.verts[0].co, e.verts[1].co )):
				addEdge = 1
		edgeCount += addEdge
	#print( " edgeCount : ", edgeCount )
	bpy.ops.object.mode_set(mode = 'OBJECT')
	# free and prevent further access
	bm.free()
	
	return edgeCount
	

	
# Counting the faces for a given object in Camera view 
# and Count materials and textures of each face in camera view as well.
def CountFaceForObject( scene, obj, cam, mList, tList ):
	camSetting = CameraSetting( scene, cam, obj )
	
	# Enter EDIT mode of this object since bmesh only can be acess in EDIT mode
	if obj.type == 'MESH':
		scene.objects.active = obj
		bpy.ops.object.mode_set(mode = 'EDIT')
	
	me = obj.data
	# Acess bmesh of this mesh
	if me.is_editmode:
		bm = bmesh.from_edit_mesh(me)
	else:
		bm = bmesh.new()
		bm.from_mesh(me)
		
	faceCount = 0
	area = 0.0
	minMaxOfXYOfAll= []
	fi = 0
	# Go through all polygons
	for f in bm.faces:
		addFace = 0
		# If one of the collected vertices on the selected face is in camera view border, 
		# then count the face. Otherwise, don't count it.
		for v in f.verts :
			x, y, z = camSetting.CalcProjected( v.co )
			if (0 <= x <= 1 and 0 <= y <= 1 and cam.data.clip_start <= z <= cam.data.clip_end) :
				addFace = 1
				break
		# Check whether exist one of edges in the face will across the camera view border, 
		# if yes, count the face. Otherwise, don't count it.
		# Only check the edge whose endpoints are vertice of the face
		if addFace == 0:
			for e in f.edges:
				if (camSetting.edgeCheckAcross( cam, e.verts[0].co, e.verts[1].co )):
					addFace = 1
					break
		if addFace == 1:
			# Count the faces
			faceCount += addFace
			# Calaulate the area
			area = area + f.calc_area()
			# If face in the Camera view, Count materials and textures as well.
			#print(f.material_index)
			if len(obj.material_slots) != 0:
				mat = obj.material_slots[f.material_index].material
				# Check there is a material in this material slot 
				# And if the material hasn't be counted ever, then Count it
				if mat and mat not in mList:
					mList.append(mat)
					if mat.use_nodes:
						# go through all nodes
						for n in mat.node_tree.nodes:
							# Check the type of texture
							if n.type == 'TEX_IMAGE' and n.image.name not in tList:
								tList.append(n.image.name)
								#print('Face', f.index, 'uses', n.image.name)
					# go through all texture slots in the material slot
					else:
						for mtext_slot in mat.texture_slots:	
							# Check the texture slot is not None
							if mtext_slot is not None:
								text = mtext_slot.texture
								# Check there is a texture in this texture slot 
								# And if the texture hasn't be counted ever, then Count it
								if text and text not in tList:
									tList.append(text)
	bpy.ops.object.mode_set(mode = 'OBJECT')
	# free and prevent further access
	bm.free()
	
	return ( faceCount, mList, tList, area, minMaxOfXYOfAll )


	
# Project the vertices of faces to the end plane of camera, 
# and find the minimum and maximum of x and y
def ProjVertAndFindMinMaxXY( scene, cam, obj, proj_v_list, minMaxOfXYOfAll):
	camSetting = CameraSetting( scene, cam, obj )
	min_max_Of_xy = camSetting.ProjVertToEndPlaneFindRec( cam, proj_v_list )
	# Initiate the values of min_max_Of_xy_of_all and the vertices of vListInCam
	if minMaxOfXYOfAll == []:
		minMaxOfXYOfAll = min_max_Of_xy
	else:
		if min_max_Of_xy[0] < minMaxOfXYOfAll[0]:
			minMaxOfXYOfAll[0] = min_max_Of_xy[0]
		elif min_max_Of_xy[1] > minMaxOfXYOfAll[1]:
			minMaxOfXYOfAll[1] = min_max_Of_xy[1]
		if min_max_Of_xy[2] < minMaxOfXYOfAll[2]:
			minMaxOfXYOfAll[2] = min_max_Of_xy[2]
		elif min_max_Of_xy[3] > minMaxOfXYOfAll[3]:
			minMaxOfXYOfAll[3] = min_max_Of_xy[3]
	
	return minMaxOfXYOfAll


	
#Counting materials and textures of a given object in the scene
def CountMatTexForObjectInScene( obj, mList, tList ):
	for ms in obj.material_slots:
		mat = ms.material
		# Check there is a material in this material slot 
		# And if the material hasn't be counted ever, then Count it
		if mat and mat not in mList:
			mList.append(mat)
			if mat.use_nodes:
				for n in mat.node_tree.nodes:
					# Check the type of texture
					if n.type == 'TEX_IMAGE' and n.image.name not in tList:
						tList.append(n.image.name)
			# go through all texture slots in the material slot
			else:
				for text_slot in mat.texture_slots:	
					# Check the texture slot is not None
					if text_slot is not None:
						text = text_slot.texture
						# Check there is a texture in this texture slot 
						# And if the texture hasn't be counted ever, then Count it
						if text and text not in tList:
							tList.append(text)
	return ( mList, tList )



# duplicate objects
def DuplicObj( scene, cam, obj, objList, times ):
	camSetting = CameraSetting( scene, cam, obj )
	# get the size of all objects
	minMaxOfXYOfAll = []
	for obji in objList:
		#proj_v_list = []
		if (hasattr(obji.data, "vertices")): 
			proj_v_list = [(v.co.x, v.co.y, v.co.z) for v in obji.data.vertices]
			minMaxOfXYOfAll = ProjVertAndFindMinMaxXY( scene, cam, obj, proj_v_list, minMaxOfXYOfAll)
		# if lights are in the camera view, duplicate them, too
		elif (hasattr(obji, "location") and obji.type != 'camera'):
			# check if the light is in camera view		
			x, y, z = camSetting.CalcProjected( obji.location )
			if(0 <= x <= 1 and 0 <= y <= 1 and cam.data.clip_start <= z <= cam.data.clip_end):
				proj_v_list = [(obji.location[0], obji.location[1], obji.location[2])]
				minMaxOfXYOfAll = ProjVertAndFindMinMaxXY( scene, cam, obj, proj_v_list, minMaxOfXYOfAll)
	# get the width and height of end plane of camera view
	width = int( camSetting.deltaX * cam.data.clip_end )
	Height = int( camSetting.deltaY * cam.data.clip_end )
	
	random_field_x = width - int( minMaxOfXYOfAll[1] - minMaxOfXYOfAll[0] )
	random_field_y = Height - int( minMaxOfXYOfAll[3] - minMaxOfXYOfAll[2] )
	for i in range(0, times):
		# randomly choose the new position
		random_x = random.randint(-abs(int(random_field_x/2)), abs(int(random_field_x/2)))
		random_y = random.randint(-abs(int(random_field_y/2)), abs(int(random_field_y/2)))
		ri = 0
		# check if the position is in the range or not
		while ( ri == 0 or (random_x == 0 and random_y == 0) ):
			if ri == 0:
				ri += 1
			while ( (minMaxOfXYOfAll[0] + random_x) < (camSetting.minX * cam.data.clip_end) or 
					(minMaxOfXYOfAll[1] + random_x) > (camSetting.maxX * cam.data.clip_end) ):
				random_x = random.randint(-abs(int(random_field_x/2)), abs(int(random_field_x/2)))
			while ( (minMaxOfXYOfAll[2] + random_y) < (camSetting.minY * cam.data.clip_end) or 
					(minMaxOfXYOfAll[3] + random_y) > (camSetting.maxY * cam.data.clip_end) ):
				random_y = random.randint(-abs(int(random_field_y/2)), abs(int(random_field_y/2)))
		
		random_z = random.randint( -int(cam.data.clip_end) / 5, int(cam.data.clip_end) / 5 )
		while( random_z ==0 ):
			random_z = random.randint( -int(cam.data.clip_end) / 5, int(cam.data.clip_end) / 5 )
		for obji in objList:
			if (hasattr(obji.data, "vertices") and (obji.type != 'CAMERA') and (obji is not None) ):
				new_obj = obji.copy()
				new_obj.data = obji.data.copy()
				for v in new_obj.data.vertices:
					# proj to end plane
					# Object coordinate in camera view
					camCo = camSetting.toCameraMatrix * v.co
					# Z is "inverted" as camera view is pointing to -Z of the camera
					oldDistance = -camCo.z
					# the coordinate of the projected vertice
					if(oldDistance != 0):
						x = camCo.x * cam.data.clip_end / oldDistance
						y = camCo.y * cam.data.clip_end / oldDistance
					else:
						x = camCo.x * cam.data.clip_end / 1
						y = camCo.y * cam.data.clip_end / 1
					new_z = -oldDistance + random_z
					new_x = (x+random_x) / int(cam.data.clip_end) * abs(new_z)
					new_y = (y+random_y) / int(cam.data.clip_end) * abs(new_z)
					newCamCo = Vector((new_x, new_y, -abs(new_z)))
					# Transform Camera coordinate to object coordinate
					v.co = camSetting.toObjectMatrix * newCamCo
				scene.objects.link(new_obj)
				
			
			elif (hasattr(obji, "location") and (obji.type != 'CAMERA') and (obji is not None) ):
				new_obj = obji.copy()
				new_obj.data = obji.data.copy()
				
				# proj to end plane
				# Object coordinate in camera view
				camCo = camSetting.toCameraMatrix * new_obj.location
				# Z is "inverted" as camera view is pointing to -Z of the camera
				oldDistance = -camCo.z
				# the coordinate of the projected vertice
				if(oldDistance != 0):
					x = camCo.x * cam.data.clip_end / oldDistance
					y = camCo.y * cam.data.clip_end / oldDistance
				else:
					x = camCo.x * cam.data.clip_end / 1
					y = camCo.y * cam.data.clip_end / 1
				new_z = -oldDistance + random_z
				new_x = (x+random_x) / int(cam.data.clip_end) * abs(new_z)
				new_y = (y+random_y) / int(cam.data.clip_end) * abs(new_z)
				newCamCo = Vector( (new_x, new_y, -abs(new_z)) )
				# Transform Camera coordinate to object coordinate
				new_obj.location = camSetting.toObjectMatrix * newCamCo
				scene.objects.link(new_obj)
					
				
			elif (obji.type != 'CAMERA' ):
				#print(obji, "cannot be move")
				obji.select = False
			

'''			
def qhull(sample):
    link = lambda a,b: numpy.concatenate((a,b[1:]))
    edge = lambda a,b: numpy.concatenate(([a],[b]))

    def dome(sample,base): 
        h, t = base
        dists = numpy.dot(sample-h, numpy.dot(((0,-1),(1,0)),(t-h)))
        outer = numpy.repeat(sample, dists>0, axis=0)
        
        if len(outer):
            pivot = sample[numpy.argmax(dists)]
            return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
        else:
            return base

    if len(sample) > 2:
        axis = sample[:,0]
        base = numpy.take(sample, [numpy.argmin(axis), numpy.argmax(axis)], axis=0)
        return link(dome(sample, base),
                    dome(sample, base[::-1]))
    else:
        return sample
'''			
			

def CreatFace ( scene, vList ):
	faces = []
	face = []
	face.extend( [ i for i in range(0, len(vList)) ] )
	#print("face: ", face)
	#tuple(face)
	faces.append(face)
	mesh_data = bpy.data.meshes.new("end_plane")
	mesh_data.from_pydata(vList, [], faces)
	mesh_data.update()

	obj = bpy.data.objects.new("End_plane", mesh_data)

	scene = bpy.context.scene
	scene.objects.link(obj)
	obj.select = True

	
	
	
##--------------------Main Function--------------------##

def main():
	# Decode the arguement of the input
	while True: 
		# Invalid input
		#if len(sys.argv) < 10 or len(sys.argv) > 11: #for windows, mac
		if len(sys.argv) < 12 or len(sys.argv) > 13: #for linux 
			print ("Please input the following information about the .blend file after .py: ")
			print ("path name server render_engine resolution incrFOV duplication ") 
			print ("Ps. path is relative to the current path")
			print ("Ps. name didn't include .blend")
			print ("Ps. render_engine is either BLENDER_RENDER or CYCLES")
			print ("Ps. resolution is percentage of resolution")
			sys.exit()
		# the .blend file is under the current path
		#elif len(sys.argv) == 10: #for windows, mac
		elif len(sys.argv) == 12: #for linux
			path_name = ''
			#file_name = sys.argv[4] #for windows, mac
			file_name = sys.argv[6] #for linux
			#servers = sys.argv[5] #for windows, mac
			servers = sys.argv[7] #for linux
			#render_engine = sys.argv[6] #for windows, mac
			render_engine = sys.argv[8] #for linux
			#persentResolOfCam = sys.argv[7] #for windows, mac
			persentResolOfCam = sys.argv[9] #for linux
			#angle_of_cameara = sys.argv[8] #for windows, mac
			angle_of_cameara = sys.argv[10] #for linux
			#duplication = sys.argv[9] #for windows, mac
			duplication = sys.argv[11] #for linux
			break
		# the .blend file is not under the current path 
		else :
			# path is relative to the current path
			#path_name = sys.argv[4] + "/" #for windows, mac
			path_name = sys.argv[6] + "/" #for linux
			#file_name = sys.argv[5] #for windows, mac
			file_name = sys.argv[7] #for linux
			#servers = sys.argv[6] #for windows, mac
			servers = sys.argv[8] #for linux
			#render_engine = sys.argv[7] #for windows, mac
			render_engine = sys.argv[9] #for linux
			#persentResolOfCam = sys.argv[8] #for windows, mac
			persentResolOfCam = sys.argv[10] #for linux
			#angle_of_cameara = sys.argv[9] #for windows, mac
			angle_of_cameara = sys.argv[11] #for linux
			#duplication = sys.argv[10] #for windows, mac
			duplication = sys.argv[12] #for linux
			break
	'''
	for i in range(0, len(sys.argv)):	
		print("sys.argv[", i, "]: ", sys.argv[i])
	'''

	#print state
	print("Read scene,  ", datetime.datetime.now())
	# Open blender file
	file_path_name = path_name + file_name + ".blend"
	bpy.ops.wm.open_mainfile(filepath = file_path_name)
	
	# print scene infomation
	print("File: ", file_name)
	print("Solution: ", persentResolOfCam)
	print("Increase FOV: ", angle_of_cameara)
	print("Duplica: ", duplication)
	print("Engine: ", render_engine)
	
	#print state
	print("Set render engine,  ", datetime.datetime.now())
	# Set render engine
	if ( render_engine != "BLENDER_RENDER" ) and ( render_engine != "CYCLES" ):
		print( "render_engine is either BLENDER_RENDER or CYCLES" )
		sys.exit()
	bpy.context.scene.render.engine = render_engine
	
	# Setting scene and camera
	scene = bpy.context.scene
	cam = scene.objects['Camera']
	
	#print state
	print("Set resolution,  ", datetime.datetime.now())
	# Setting resolution of RenderResult
	scene.render.resolution_percentage = float(persentResolOfCam)
	scene.render.use_border = False
	
	# All objects in Scene
	objects_in_scene = scene.objects
	
	#print state
	print("Set FOV,  ", datetime.datetime.now())
	# Setting the angle of Camera
	cam.data.angle += float(angle_of_cameara)
	#print("FOV", cam.data.angle)
	
	#print state
	print("duplicate objects,  ", datetime.datetime.now())
	# duplicate objects
	# Randomly choose an object
	if int(duplication) > 0:
		for obj in objects_in_scene:
			if (hasattr(obj.data, "vertices")):
				tempObj = obj
				break
		DuplicObj( scene, cam, obj, objects_in_scene, int(duplication) )
	
	
	#print state
	print("Get features in the scene,  ", datetime.datetime.now())
	# Count vertices, edges, faces in the whole scene
	# Count lights and each type of light(point, sun, spot, hemi, area).
	# Count materials and textures in whole scene as well.
	numVertsInScene = 0
	numEdgesInScene = 0
	numFacesInScene = 0
	num_of_lights = 0
	num_of_point_lights = 0
	num_of_sun_lights = 0
	num_of_spot_lights = 0
	num_of_hemi_lights = 0
	num_of_area_lights = 0
	matListInScene = []
	texListInScene = []
	TextOfLight_list = []
	# go through all objects in the Scene
	for obj in objects_in_scene :
		me = obj.data
		# Check whether VERTICEs exist and Count the vertices in given Object
		if (hasattr(me, "vertices")):
			numVertsInScene += len(me.vertices)
		# Check whether EDGEs exist and Count the vertices in given Object
		if (hasattr(me, "edges")):
			numEdgesInScene += len(me.edges)
		# Check whether FACEs exist and Count the vertices in given Object
		if (hasattr(me, "polygons")):
			numFacesInScene += len(me.polygons)
		# Count lights, each type of light(point, sun, spot, hemi, area) and textures on them.
		elif (obj.type == 'LAMP'):
			# Total number of lights
			num_of_lights +=1
			lamp = obj.data
			# go through all texture slots in this lamp
			if (hasattr(obj, "texture_slots")):
				for text_slot in obj.texture_slots:
					# Check the texture slot is not None
					if text_slot is not None:
						text = text_slot.texture
						# Check there is a texture in this texture slot 
						if text is not None:
							# If the texture hasn't be counted ever, then Count it
							if text not in TextOfLight_list:
								tList.append(text)
			# Count each types of lights
			if (lamp.type == 'POINT'):
				num_of_point_lights += 1
			elif (lamp.type == 'SUN'):
				num_of_sun_lights += 1
			elif (lamp.type == 'spot'):
				num_of_spot_lights += 1
			elif (lamp.type == 'hemi'):
				num_of_hemi_lights += 1
			else:
				num_of_area_lights += 1
		matAndTex = CountMatTexForObjectInScene(obj, matListInScene, texListInScene)
		matListInScene = matAndTex[0]
		texListInScene = matAndTex[1]
	
	
	#print state
	print("Get the feature of all objects in view, ", datetime.datetime.now())
	
	# All objects in camera view
	ocv = ObjectsInCameraView()
	objects_in_view = ocv.objects_in_view
	
	#print state
	print("Get features in the camera view, ", datetime.datetime.now())
	# Count vertices, edges, faces in camera view, 
	# And Count materials and textures when check each faces.
	numVertsInCamera = 0
	numEdgesInCamera = 0
	numFacesInCamera = 0
	matListInCamera = []
	texListInCamera = []
	totalArea = 0.0
	list_v_in_cam = []
	for obj in objects_in_scene:
		
		me = obj.data
		# Check whether VERTICEs exist 
		if (hasattr(me, "vertices")):
			# Count the vertices in given Object
			'''
			numVertsInCamera += CountVerticeForObject( scene, obj, cam )
			'''
			vertInCamera = CountVerticeForObject( scene, obj, cam )
			numVertsInCamera += vertInCamera[0]
			#print("vertInCamera[1]: ", vertInCamera[1])
			#list_v_in_cam.extend( transfVertInCamera )
			list_v_in_cam.extend( vertInCamera[1] )
		# Check whether EDGEs exist 
		if (hasattr(me, "edges")):
			# Count the edges in given Object
			numEdgesInCamera += CountEdgeForObject( scene, obj, cam )	
		# Check whether FACEs exist
		if (hasattr(me, "polygons")):
			# Count the faces in given Object
			# If face in the Camera view, Count materials and textures as well.
			faceInCameraView = CountFaceForObject( scene, obj, cam, matListInCamera, texListInCamera )
			numFacesInCamera += faceInCameraView[0]
			# If face in the Camera view, Count materials and textures as well.
			matListInCamera = faceInCameraView[1]
			texListInCamera = faceInCameraView[2]
			totalArea = totalArea + faceInCameraView[3]
	
	
	#print state
	print("Get features of width, height, pixels, ", datetime.datetime.now())
	# Calculate the width, the height and the pixels of the frame
	render = bpy.context.scene.render
	scale = render.resolution_percentage / 100
	# get width and height
	width_of_frame = render.resolution_x * scale
	height_of_frame = render.resolution_y * scale
	# get viewer pixels ( = width * height * 4 (rgba) )
	pixels_of_frame = width_of_frame * height_of_frame * 4
	
	#print state
	print("Calculate render time, ", datetime.datetime.now())
	# Calculate render time
	# set the start-time stamp
	start_render_time = datetime.datetime.now()
	# Render
	#bpy.ops.render.render()
	image_path_name = path_name + 'render_images/' + file_name + "_" + servers + "_" + render_engine + "_" + persentResolOfCam + "_" + angle_of_cameara + "_" + duplication
	bpy.context.scene.render.filepath = image_path_name
	bpy.ops.render.render(write_still = True)
	# set the end-time stamp
	end_render_time = datetime.datetime.now()
	# calculate rendering time 
	render_time = (end_render_time - start_render_time).total_seconds()

	
	# save changes to blender file
	file_path_name2 = path_name + "duplication/" + file_name + "_" + servers + "_" + render_engine + "_" + persentResolOfCam + "_" + angle_of_cameara + "_" + duplication + ".blend"
	bpy.ops.wm.save_as_mainfile(filepath=file_path_name2)
	
	
	#print state
	print("Calculate coverage, ", datetime.datetime.now())
	# Calculate coverage
	# project vertices to end plane first
	for obj in objects_in_scene:
		if (hasattr(obj.data, "vertices")):
			tempObj = obj
			break
	temp_camSetting = CameraSetting( scene, cam, tempObj )
	v_to_calc_convex_hull = []
	for v in list_v_in_cam:
		v0_camSetting = CameraSetting( scene, cam, v[0] )
		temp_proj = v0_camSetting.ProjVertToEndPlane( cam, [v[1]] )
		#print("temp_proj: ", temp_proj)
		v_to_calc_convex_hull.extend(temp_proj)
		del v0_camSetting
	
	# Quick Hull
	CalcCover = CalcCoverage()
	v_of_convex_hull = CalcCover.get_hull_points(v_to_calc_convex_hull)
	
	'''
	# Quick Hull
	v_array_to_calc_convex_hull = numpy.array(v_to_calc_convex_hull)
	v_of_convex_hull = qhull(v_array_to_calc_convex_hull)
	'''
	
	# Create a mesh and an object for cover faces 
	coverName = 'coverObj'
	coverMe = bpy.data.meshes.new('End_plane')
	coverObj = bpy.data.objects.new(coverName, coverMe)
	coverObj.show_name = True
	# Link object to scene
	bpy.context.scene.objects.link(coverObj)
	
	# Transform Camera coordinate to object coordinate
	end_plane_verts = []
	for vc in v_of_convex_hull:
		newCamCo = Vector( (vc[0], vc[1], -cam.data.clip_end) )
		new_camSetting = CameraSetting( scene, cam, coverObj )
		new_v = new_camSetting.toObjectMatrix * newCamCo
		end_plane_verts.append( new_v )
		
	
	# Create face of Quick Hull
	CreatFace( scene, end_plane_verts )
	proj_end_plane = scene.objects["End_plane"]
	end_plane_area = temp_camSetting.deltaX * temp_camSetting.deltaY * cam.data.clip_end * cam.data.clip_end
	
	# coverage
	proj_end_plane_area = 0
	for p in proj_end_plane.data.polygons:
		proj_end_plane_area += p.area
	coverage = proj_end_plane_area / end_plane_area
	#print(coverage)
	
	
	#print state
	print("Output features into CSV file, ", datetime.datetime.now())
	# Output features into CSV file
	# check csv file exists or not
	if (os.path.isfile("features.csv") == False):
		# Write title
		csvfile = open("features.csv","w", newline='')
		writecsv = csv.writer(csvfile)
		writecsv = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
		#title = [ ["File"] + ["Percentage of resolution"] 
		title = [ ["File"] + ["Server"] + ["Render engine"]
				+ ["Percentage of resolution"] + ["increass FOV of camera"]  + ["Duplication"]  
				+ ["objects in the Scene"] + ["Vertices in the Scene"] + ["Edges in the Scene"] 
				+ ["Faces in the Scene"] + ["Lights"] + ["Point lights"] 
				+ ["Sun lights"] + ["Spot lights"] + ["Hemi lights"]
				+ ["Area lights"] + ["Texeures of lights"] + ["Materials in the Scene"] 
				#+ ["Texeures in the Scene"] + ["Vertices in Camera view"] 
				+ ["Texeures in the Scene"] + ["Objects in Camera view"] + ["Vertices in Camera view"] 
				+ ["Edges in Camera view"] + ["Faces in Camera view"]+ ["Materials in Camera view"] 
				+ ["Textures in Camera view"] + ["Area of face"] + ["Width"]
				+ ["Height"] + ["pixels"] + ["Render time"] + ["Coverage"] ]
		writecsv.writerows(title)
		csvfile.close()
	# Write features of every file
	csvfile = open("features.csv","a", newline='')
	writecsv = csv.writer(csvfile)
	writecsv = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
	#datas = [ [file_name] + [float(persentResolOfCam)]
	datas = [ [file_name] + [servers] + [render_engine] 
			+ [float(persentResolOfCam)] + [float(angle_of_cameara)]  + [int(duplication)]
			+ [len(objects_in_scene)] + [numVertsInScene] + [numEdgesInScene]
			+ [numFacesInScene] + [num_of_lights] + [num_of_point_lights]
			+ [num_of_sun_lights] + [num_of_spot_lights] + [num_of_hemi_lights]
			+ [num_of_area_lights] + [len(TextOfLight_list)] + [len(matListInScene)]
			#+ [len(texListInScene)] + [numVertsInCamera]
			+ [len(texListInScene)] + [len(objects_in_view)] + [numVertsInCamera]
			+ [numEdgesInCamera] + [numFacesInCamera] + [len(matListInCamera)]
			+ [len(texListInCamera)] + [totalArea] + [width_of_frame] 
			+ [height_of_frame] + [pixels_of_frame] + [render_time] + [coverage] ]
	writecsv.writerows(datas)
	# Close file
	csvfile.close()
	
	'''
	# save changes to blender file
	file_path_name2 = path_name + "/" + file_name + "du.blend"
	bpy.ops.wm.save_as_mainfile(filepath=file_path_name2)
	'''
	
	

if __name__ == "__main__":main()
