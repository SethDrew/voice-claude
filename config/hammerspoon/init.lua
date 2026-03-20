
-- voice-router hotkey — press Right Option to record, release to stop
-- Uses warm listen daemon for instant mic activation
local voiceRouter = {}
voiceRouter.optDown = false
voiceRouter.stateFile = os.getenv("HOME") .. "/.local/share/voice-router/daemon-state.json"
voiceRouter.resultsFile = os.getenv("HOME") .. "/.local/share/voice-router/daemon-results.jsonl"
voiceRouter.pidFile = os.getenv("HOME") .. "/.local/share/voice-router/listen-daemon.pid"
voiceRouter.pollTimer = nil
voiceRouter.routeQueue = {}
voiceRouter.routing = false
voiceRouter.lastResultLine = 0
voiceRouter.pendingCount = 0  -- how many recordings are still being transcribed
voiceRouter.micAlert = nil

local function getDaemonPid()
    local f = io.open(voiceRouter.pidFile, "r")
    if not f then return nil end
    local pid = f:read("*a"):match("^%s*(.-)%s*$")
    f:close()
    if pid and #pid > 0 then return pid end
    return nil
end

local function signalDaemon(sig)
    local pid = getDaemonPid()
    if pid then
        -- Check process is alive before signaling
        local _, status = hs.execute("kill -0 " .. pid .. " 2>/dev/null")
        if not status then return false end
        hs.execute("kill -" .. sig .. " " .. pid)
        return true
    end
    return false
end

local function processQueue()
    if voiceRouter.routing or #voiceRouter.routeQueue == 0 then return end

    voiceRouter.routing = true
    local text = table.remove(voiceRouter.routeQueue, 1)

    if not text or #text == 0 then
        voiceRouter.routing = false
        processQueue()
        return
    end

    local env = {
        PATH = os.getenv("HOME") .. "/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin",
        HOME = os.getenv("HOME"),
    }

    local task = hs.task.new(
        os.getenv("HOME") .. "/.local/bin/voice-route",
        function(exitCode, stdout, stderr)
            voiceRouter.routing = false
            processQueue()
        end,
        {"--text", text}
    )
    task:setEnvironment(env)
    task:start()
end

local function consumeResults()
    local f = io.open(voiceRouter.resultsFile, "r")
    if not f then return end

    local lineNum = 0
    for line in f:lines() do
        lineNum = lineNum + 1
        if lineNum > voiceRouter.lastResultLine then
            local ok, result = pcall(hs.json.decode, line)
            if ok and result and result.text and #result.text > 0 then
                table.insert(voiceRouter.routeQueue, result.text)
                voiceRouter.pendingCount = math.max(0, voiceRouter.pendingCount - 1)
            elseif ok then
                -- empty transcription, still decrement pending
                voiceRouter.pendingCount = math.max(0, voiceRouter.pendingCount - 1)
            end
            voiceRouter.lastResultLine = lineNum
        end
    end
    f:close()

    processQueue()
end

local function ensurePolling()
    if voiceRouter.pollTimer then return end
    voiceRouter.pollTimer = hs.timer.doEvery(0.15, function()
        consumeResults()

        -- Stop when nothing pending and queue drained
        if voiceRouter.pendingCount <= 0 and #voiceRouter.routeQueue == 0
           and not voiceRouter.routing and not voiceRouter.optDown then
            -- Truncate results file to prevent unbounded growth
            local f = io.open(voiceRouter.resultsFile, "w")
            if f then f:close() end
            voiceRouter.lastResultLine = 0
            voiceRouter.pollTimer:stop()
            voiceRouter.pollTimer = nil
        end
    end)
end

voiceRouter.tap = hs.eventtap.new({hs.eventtap.event.types.flagsChanged}, function(event)
    local flags = event:getRawEventData().CGEventData.flags
    local rightOpt = (flags & 0x40) ~= 0 and (flags & 0x20) == 0
    local ctrlPressed = (flags & 0x40000) ~= 0
    local cmdPressed = (flags & 0x100000) ~= 0
    local shiftPressed = (flags & 0x20000) ~= 0

    local optOnly = rightOpt and not ctrlPressed and not cmdPressed and not shiftPressed

    if optOnly and not voiceRouter.optDown then
        voiceRouter.optDown = true
        signalDaemon("USR1")
        -- Show mic indicator (stays until we close it on release)
        hs.alert.closeAll()
        voiceRouter.micAlert = hs.alert.show("🎤", nil, nil, 60)

    elseif not rightOpt and voiceRouter.optDown then
        voiceRouter.optDown = false
        signalDaemon("USR2")
        hs.alert.closeAll()
        voiceRouter.pendingCount = voiceRouter.pendingCount + 1
        ensurePolling()
    end

    return false
end)

voiceRouter.tap:start()
