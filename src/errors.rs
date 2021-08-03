/// An error which may occur during WFC
pub enum WfcError {
    /// An algorithm failed to find a solution
    TooManyContradictions,
    /// We don't know what it is but it seems terrific
    SomeCreepyShit
}
