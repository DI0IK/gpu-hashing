use lazy_static::lazy_static;
use std::sync::mpsc::Sender;
use std::sync::Mutex;

lazy_static! {
    static ref EVENT_SENDER: Mutex<Option<Sender<String>>> = Mutex::new(None);
}

pub fn init_event_sender(s: Sender<String>) {
    if let Ok(mut g) = EVENT_SENDER.lock() {
        *g = Some(s);
    }
}

pub fn publish_event(msg: &str) {
    if let Ok(g) = EVENT_SENDER.lock() {
        if let Some(tx) = &*g {
            let _ = tx.send(msg.to_string());
        }
    }
}
