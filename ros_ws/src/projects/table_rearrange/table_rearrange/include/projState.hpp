#ifndef TABLE_REARRANGE_STATE_H
#define TABLE_REARRANGE_STATE_H

// Forward declaration
class TableRearrangeState;

class TableRearrange {
public:
  TableRearrange();
  ~TableRearrange();

private:
  friend class TableRearrangeState;
  void changeState(TableRearrangeState* nState); // Allow state transitions

  TableRearrangeState* _state;	// Private state
private:			// Invariant stuff for _state
  bool hasRefPicture;
  bool hasDiffPicture;
  bool synthesizedGrasp;
  bool objectHeld;
};

// TODO: Implement default behavior
class TableRearrangeState {
public:
  TableRearrangeState();
  ~TableRearrangeState();
  
  virtual void moveToStart();
  virtual void moveToPose();
  virtual void takeReferencePicture();
  virtual void takeDiffPicture();
  virtual void randomizeObjects();
  virtual void synthGrasp();
  virtual void moveToPose();
  virtual void rearrangeHeld();

protected:
  void changeState(TableRearrange* ctxt, TableRearrangeState* nState);
};

// Position-related states
class DefaultState : public TableRearrangeState {};
class AtStartState : public TableRearrangeState {};
class AtPoseState : public TableRearrangeState {};

// Grasp-related states
class BeforeGraspState : public TableRearrangeState {};
class AtGraspState : public TableRearrangeState {};
class AfterGraspState : public TableRearrangeState {};

// Manipulation-related states
class BeforePlacingState : public TableRearrangeState {};
class AtPlacingState : public TableRearrangeState {};
class AfterPlacingState : public TableRearrangeState{};

#endif
