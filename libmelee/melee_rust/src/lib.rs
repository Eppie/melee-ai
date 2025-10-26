use pyo3::prelude::*;

mod enums;
mod gamestate;
mod parser;

use enums::*;
use gamestate::*;
use parser::SlpParser;

/// A Python module implemented in Rust.
#[pymodule]
fn melee_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Register enums
    m.add_class::<Stage>()?;
    m.add_class::<Menu>()?;
    m.add_class::<SubMenu>()?;
    m.add_class::<Character>()?;
    m.add_class::<Action>()?;
    m.add_class::<Button>()?;
    m.add_class::<ProjectileType>()?;
    
    // Register gamestate types
    m.add_class::<Position>()?;
    m.add_class::<ECB>()?;
    m.add_class::<ControllerState>()?;
    m.add_class::<PlayerState>()?;
    m.add_class::<Projectile>()?;
    m.add_class::<GameState>()?;
    
    // Register parser
    m.add_class::<SlpParser>()?;
    
    Ok(())
}

