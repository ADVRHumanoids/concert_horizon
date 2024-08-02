import rospy
from modes import OperationMode
from std_srvs.srv import SetBool, SetBoolRequest

class ModeHandler:
    def __init__(self, controller):

        self.teach_mode_service = None
        self.follow_me_mode_service = None

        self.operation_mode = OperationMode.IDLE

        self.init_services()

        # class is required to have getMode and setMode
        self.controller = controller


    def init_services(self):

        print('Opening services for mode of operation...\n')
        # driving mode
        self.follow_me_mode_service = rospy.Service('/force_mpc/operation_mode/follow_me/switch', SetBool, self.set_follow_me_mode)
        # teaching mode
        # self.teach_mode_service = rospy.Service('/force_mpc/operation_mode/teach/switch', SetBool, self.set_teach_mode)
        # hybrid mode
        # self.hybrid_mode_service = rospy.Service('/force_mpc/operation_mode/hybrid/switch', SetBool, self.set_hybrid_mode)
        # homing mode
        # self.homing_mode_service = rospy.Service('/force_mpc/operation_mode/homing/switch', SetBool, self.set_homing_mode)

        print("done.\n")


    def set_teach_mode(self, req):
        if req.data:
            self.controller.setMode(OperationMode.TEACH)
        else:
            if self.controller.getMode() == OperationMode.TEACH:
                self.controller.setMode(OperationMode.IDLE)

        return {'success': True}

    def set_follow_me_mode(self, req):
        if req.data:
            self.controller.setMode(OperationMode.FOLLOW_ME)
        else:
            if self.controller.getMode() == OperationMode.FOLLOW_ME:
                self.controller.setMode(OperationMode.IDLE)

        return {'success': True}

    def set_hybrid_mode(self, req):
        if req.data:
            self.controller.setMode(OperationMode.HYBRID)
        else:
            if self.controller.getMode() == OperationMode.HYBRID:
                self.controller.setMode(OperationMode.IDLE)

        return {'success': True}

    def set_homing_mode(self, req):
        if req.data:
            self.controller.setMode(OperationMode.HOMING)
        else:
            if self.controller.getMode() == OperationMode.HOMING:
                self.controller.setMode(OperationMode.IDLE)

        return {'success': True}

    def getMode(self):

        return self.operation_mode
