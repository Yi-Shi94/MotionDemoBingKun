
import os

import fbx
import FbxCommon


class FBX_Class(object):

    def __init__(self, filename):
        """
        FBX Scene Object
        """
        if not os.path.exists(filename):
            error_msg = '"{}" filepath is invalid or does not exist.\n'.format(filename)
            raise Exception(error_msg)
        self.version = __version__
        self.filename = filename        
        self.scene = None
        self.sdk_manager = None
        self.sdk_manager, self.scene = FbxCommon.InitializeSdkObjects()
        try:
            FbxCommon.LoadScene(self.sdk_manager, self.scene, filename)
        except OSError, e:
            if e.errno != os.errno.EEXIST:
                raise   
            # time.sleep might help here
            pass
        self.root_node = self.scene.GetRootNode()
        self.scene_nodes = self.get_scene_nodes()

    def close(self):
        """
        You need to run this to close the FBX scene safely
        """
        # destroy objects created by the sdk
        self.sdk_manager.Destroy()

    def __get_scene_nodes_recursive(self, node):
        """
        Rescursive method to get all scene nodes
        this should be private, called by get_scene_nodes()
        """
        self.scene_nodes.append(node)
        for i in range(node.GetChildCount()):
            self.__get_scene_nodes_recursive(node.GetChild(i))

    def __cast_property_type(self, fbx_property):
        """
        Cast a property to type to properly get the value
        """
        casted_property = None

        unsupported_types = [fbx.eFbxUndefined, fbx.eFbxChar, fbx.eFbxUChar, fbx.eFbxShort, fbx.eFbxUShort, fbx.eFbxUInt,
                             fbx.eFbxLongLong, fbx.eFbxHalfFloat, fbx.eFbxDouble4x4, fbx.eFbxEnum, fbx.eFbxTime,
                             fbx.eFbxReference, fbx.eFbxBlob, fbx.eFbxDistance, fbx.eFbxDateTime, fbx.eFbxTypeCount]

        # property is not supported or mapped yet
        property_type = fbx_property.GetPropertyDataType().GetType()
        if property_type in unsupported_types:
            return None
        
        if property_type == fbx.eFbxBool:
            casted_property = fbx.FbxPropertyBool1( fbx_property )
        elif property_type == fbx.eFbxDouble:
            casted_property = fbx.FbxPropertyDouble1( fbx_property )
        elif property_type == fbx.eFbxDouble2:
            casted_property = fbx.FbxPropertyDouble2( fbx_property )
        elif property_type == fbx.eFbxDouble3:
            casted_property = fbx.FbxPropertyDouble3( fbx_property )
        elif property_type == fbx.eFbxDouble4:
            casted_property = fbx.FbxPropertyDouble4( fbx_property )
        elif property_type == fbx.eFbxInt:
            casted_property = fbx.FbxPropertyInteger1( fbx_property )
        elif property_type == fbx.eFbxFloat:
            casted_property = fbx.FbxPropertyFloat1( fbx_property )
        elif property_type == fbx.eFbxString:
            casted_property = fbx.FbxPropertyString( fbx_property )
        else:
            raise ValueError( 'Unknown property type: {0} {1}'.format(property.GetPropertyDataType().GetName(), property_type))

        return casted_property

    def get_scene_nodes(self):
        """
        Get all nodes in the fbx scene      
        """
        self.scene_nodes = []
        for i in range(self.root_node.GetChildCount()):
            self.__get_scene_nodes_recursive(self.root_node.GetChild(i))
        return self.scene_nodes

    def get_type_nodes(self, type):
        """
        Get nodes from the scene with the given type
        display_layer_nodes = fbx_file.get_type_nodes( u'DisplayLayer' )
        skeleton_nodes = fbx_file.get_type_nodes( u'LimbNodes' )
        """     
        nodes = []
        num_objects = self.scene.RootProperty.GetSrcObjectCount()
        for i in range(0, num_objects):
            node = self.scene.RootProperty.GetSrcObject(i)
            if node:
                if node.GetTypeName() == type:
                    nodes.append(node)      
        return nodes

    def get_class_nodes(self, class_id):
        """
        Get nodes in the scene with the given classid
        geometry_nodes = fbx_file.get_class_nodes( fbx.FbxGeometry.ClassId )
        skeleton_nodes = fbx_file.get_class_nodes( fbx.FbxSkeleton.ClassId ) # 'FbxSkeleton'
        """
        nodes = []
        num_objects = self.scene.RootProperty.GetSrcObjectCount()
        num_nodes = self.scene.RootProperty.GetSrcObjectCount(fbx.FbxCriteria.ObjectType(class_id))
        # for index in range(0, num_objects):
        for index in range(0, num_nodes):
            node = self.scene.RootProperty.GetSrcObject(fbx.FbxCriteria.ObjectType(class_id), index)
            if node:
                nodes.append(node)                  
        return nodes

    def get_joints(self, classNames=None):
        """
        Get joints from fbx file.
        """
        classNames = classNames or ['FbxSkeleton']

        if not isinstance(classNames, (list, tuple)):
            classNames = list(classNames)
        # SKM fbx_file
        # fbx_filepaths = [r'P:\GreenPasturesDev\Source\Characters\AI\Sentinel\Twisted_Sentinel.fbx']

        # fbx_filename = fbx_filepaths[0]
        # fbx_file = FBX_Class(fbx_filename)
        # joints = fbx_file.get_type_nodes( u'Skeleton')    
        all_nodes = self.get_scene_nodes()

        # types = list(set([node.GetTypeName() for node in all_nodes]))

        # skeletonList = [node for node in all_nodes if node.GetNodeAttribute().__class__.__name__ == 'FbxSkeleton']
        skeletonList = [node for node in all_nodes if node.GetNodeAttribute().__class__.__name__ in classNames]
        print len(skeletonList), 'skeleton nodes.'
        jointNames = [j.GetName() for j in skeletonList]

        return jointNames
    
    def get_property(self, node, property_string):
        """
        Gets a property from an Fbx node
        export_property = fbx_file.get_property(node, 'no_export')
        """     
        fbx_property = node.FindProperty(property_string)
        return fbx_property

    def get_nodes_with_property(self, property_string):
        """
        Gets an Fbx node containing property_string in it's list of property.
        nodes = fbx_file.get_nodes_with_property('no_export')
        fbx_basejnts = fbx_file.get_nodes_with_property('baseJoint')
        fbx_leafjnts = fbx_file.get_nodes_with_property('leafJoint')
        fbx_nobakejnts = fbx_file.get_nodes_with_property('noBakeJoint')
        fbx_noexport = fbx_file.get_nodes_with_property('noExport')
        """
        nodes = []
        for node in self.scene_nodes:
            fbx_property = node.FindProperty(property_string)
            if fbx_property.IsValid():
                nodes.append(node)
        nodeNames = [n.GetName() for n in nodes]

        return nodeNames

    def get_property_value(self, node, property_string):
        """
        Gets the property value from an Fbx node
        property_value = fbx_file.get_property_value(node, 'no_export')
        """ 
        fbx_property = node.FindProperty(property_string)
        if fbx_property.IsValid():
            # cast to correct property type so you can get
            casted_property = self.__cast_property_type(fbx_property)
            if casted_property:
                return casted_property.Get()
        return None
    
    def get_node_by_name(self, name, shortname=True):
        """
        Get the fbx node by name.
        
        2019.08.30
        #TODO:
            - add optional arg to return shortname, longname, namespace, etc.
        """
        self.get_scene_nodes()
        # right now this is only getting the first one found
        if shortname:
            node = [ node for node in self.scene_nodes if str(node.GetName()).split(':')[-1] == name ]
        else:
            node = [ node for node in self.scene_nodes if node.GetName() == name ]
        if node:
            return node[0]      
        return None

    def remove_namespace(self):
        """
        Remove all namespaces from all nodes
        This is not an ideal method but
        """
        self.get_scene_nodes()      
        for node in self.scene_nodes:
            orig_name = node.GetName()
            split_by_colon = orig_name.split(':')
            if len(split_by_colon) > 1:
                new_name = split_by_colon[-1:][0]
                node.SetName(new_name)
        return True

    def remove_node_property(self, node, property_string):
        """
        Remove a property from an Fbx node
        remove_property = fbx_file.remove_property(node, 'UDP3DSMAX')
        """
        node_property = self.get_property(node, property_string)
        if node_property.IsValid():
            node_property.DestroyRecursively()
            return True
        return False

    def remove_nodes_by_names(self, names):
        """
        Remove nodes from the fbx file from a list of names
        names = ['object1','shape2','joint3']
        remove_nodes = fbx_file.remove_nodes_by_names(names)
        """ 

        if names == None or len(names) == 0:
            return True

        self.get_scene_nodes()
        remove_nodes = [ node for node in self.scene_nodes if node.GetName() in names ]
        for node in remove_nodes:
            disconnect_node = self.scene.DisconnectSrcObject(node)
            remove_node = self.scene.RemoveNode(node)
        self.get_scene_nodes()
        return True

    def save(self, filename = None ):
        """
        Save the current fbx scene as the incoming filename .fbx
        """
        # save as a different filename
        if not filename is None:
            FbxCommon.SaveScene(self.sdk_manager, self.scene, filename)
        else:
            FbxCommon.SaveScene(self.sdk_manager, self.scene, self.filename)
        self.close()

    def get_matrix_from_fbxNode(self, node, pFrame=None, worldSpace=True):
        '''[summary]
        
        [description]
        
        Arguments:
            node {[type]} -- [description]
        
        Example Usage:
            # sample a frame from an fbx file and create locators to represent the skeletal pose.
            from ozPy import rigging
            animfbx = r'P:\GreenPasturesDev\Source\Characters\_Global\BaseTransformer\BaseRobot\Animation\BR_Idle_Transform_Out_Fast.fbx'
            animfbx = r'P:\GreenPasturesDev\Source\Characters\AI\Enemies\Sentinel\Animation\SENT_Com_Run_F.fbx'
            fbx_filename = animfbx
            fbx_file = FBX_Class(fbx_filename)
            all_nodes = fbx_file.get_scene_nodes()
            locs=[]
            for i, node in enumerate(all_nodes):
                name = node.GetName()
                locName = 'loc_'+ name.split(':')[-1]
                loc = rigging.makeLoc()
                locs.append(loc)
                loc.rename(locName)
                mtrxM = api.MMatrix(fbx_file.get_matrix_from_fbxNode(node, pFrame=10))
                loc.setMatrix(mtrxM, ws=1)
            pmc.select(locs, r=1)
        '''
        pFrame = pFrame or 0
        lTime = FbxCommon.FbxTime()         # The time for each key in the animation curve(s)
        lTime.SetTime(0, 0, 0, pFrame)      # 1 frame @ current frame rate
        # print lTime.GetTimeString()
        
        # if worldSpace:
        #     transformMatrix = sceneEvaluator.GetNodeGlobalTransform(node, lTime)   # GlobalMatrix(fbx.FbxAMatrix)
        # if not worldSpace:
        #     transformMatrix = sceneEvaluator.GetNodeLocalTransform(node, lTime)   # GlobalMatrix(fbx.FbxAMatrix)
        # mtrxL = []
        # for row in range(4):
        #     mtrxL = mtrxL + list(GlobalMatrix.GetRow(row))
        
        # # print i, node.GetName()
        # # print node.GetTypeName()
        
        # EvaluateGlobalTransform(...)
        #     FbxNode.EvaluateGlobalTransform(FbxTime pTime=FBXSDK_TIME_INFINITE, FbxNode.EPivotSet pPivotSet=FbxNode.eSourcePivot, bool pApplyTarget=False, bool pForceEval=False) -> FbxAMatrix
        if worldSpace:
            transformMatrix = node.EvaluateGlobalTransform(lTime)
        if not worldSpace:
            transformMatrix = node.EvaluateLocalTransform(lTime)
        # # print gTransform.Get()
        # gRot = gTransform.GetR()
        # gPos = gTransform.GetT()
        mtrxL = []
        for row in range(4):
            mtrxL = mtrxL + list(transformMatrix.GetRow(row))
        # print mtrxL

        return mtrxL
        
    def get_null_nodes(self):
        
        self.null_nodes = []
        
        for nodeIndex in range(0, self.scene.GetNodeCount()) :
            fbxNode = self.scene.GetNode(nodeIndex)
            fbxNull = fbxNode.GetNull()
            if fbxNull is not None:
                self.null_nodes.append(fbxNull)
                
        return self.null_nodes
                
    def get_mesh_nodes(self):

        self.mesh_nodes = [] 
        
        for nodeIndex in range(0, self.scene.GetNodeCount()) :
            fbxNode = self.scene.GetNode(nodeIndex)
            fbxMesh = fbxNode.GetMesh()
            if fbxMesh is not None:
                self.mesh_nodes.append(fbxMesh)
                
        return self.mesh_nodes
    
    def get_material_nodes(self):
        
        self.material_nodes = []
        
        for idx in range(0, self.scene.GetMaterialCount()) :
            fbxMat = self.scene.GetMaterial(idx)
            if fbxMat is not None:
                self.material_nodes.append(fbxMat)
                
        return self.material_nodes


    def get_start_and_end_frames(self):
        ''' Timeline default timespan '''
        lGlobalSettings = self.scene.GetGlobalSettings()
        lTs = lGlobalSettings.GetTimelineDefaultTimeSpan()
        lStart = lTs.GetStart()
        lEnd   = lTs.GetStop()
        startFrame = int(lStart.GetTimeString('', 10))
        endFrame = int(lEnd.GetTimeString('', 10))
        
        return startFrame, endFrame
        
    def get_frame_rate(self):
        lTimeModes = [ "Default Mode", "Cinema", "PAL", "Frames 30", 
                        "NTSC Drop Frame", "Frames 50", "Frames 60",
                        "Frames 100", "Frames 120", "NTSC Full Frame", 
                        "Frames 30 Drop", "Frames 1000" ] 
        timeModesCount = fbx.FbxTime.eModesCount
        # fbx.FbxTime.eDefaultMode >>> 0
        # fbx.FbxTime.eFrames30 >>> 6
                      
        lGlobalSettings = self.scene.GetGlobalSettings()
        lTimeModeIdx = lGlobalSettings.GetTimeMode()
        
        lFps = fbx.FbxTime.GetFrameRate(lTimeModeIdx)
        
        return lFps


"""
You will need to instantiate the class to access its methods
"""
#fbx_file = FBX_Class(r'd:\my_path\test.fbx')
#node = fbx_file.get_node_by_name('head')
#node_property = fbx_file.get_property(node, 'no_export')
#node_property_value = fbx_file.get_property_value( node, 'no_export')
#remove_property = fbx_file.remove_node_property(node, 'no_anim_export')
#remove_property = fbx_file.remove_node_property(node, 'no_export')
#remove_node = fbx_file.remove_nodes_by_names('hair_a_01')
#save_file = fbx_file.save(filename=r'd:\temp.fbx')